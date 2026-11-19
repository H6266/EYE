import { createHash, createHmac } from 'node:crypto';

interface S3ImageStoreOptions {
  env?: NodeJS.ProcessEnv;
  fetcher?: typeof fetch;
}

export class S3ImageStore {
  private readonly env: NodeJS.ProcessEnv;
  private readonly fetcher: typeof fetch;

  constructor(options: S3ImageStoreOptions = {}) {
    this.env = options.env ?? process.env;
    this.fetcher = options.fetcher ?? fetch;
  }

  isConfigured(): boolean {
    return Boolean(
      this.env.S3_BUCKET ||
        this.env.S3_ACCESS_KEY_ID ||
        this.env.S3_SECRET_ACCESS_KEY ||
        this.env.S3_ENDPOINT_URL ||
        this.env.S3_PUBLIC_BASE_URL
    );
  }

  async putResult(
    jobId: string,
    contentType: string,
    bytes: Uint8Array
  ): Promise<string | undefined> {
    if (!this.isConfigured()) return undefined;
    const bucket = this.env.S3_BUCKET;
    const accessKey = this.env.S3_ACCESS_KEY_ID;
    const secretKey = this.env.S3_SECRET_ACCESS_KEY;
    if (!bucket || !accessKey || !secretKey) {
      throw new Error(
        'S3 storage requires S3_BUCKET, S3_ACCESS_KEY_ID, and S3_SECRET_ACCESS_KEY.'
      );
    }

    const region = this.env.S3_REGION || 'us-east-1';
    const endpoint = this.endpoint(region);
    const key = `auteur-studio/${jobId}.${extensionFor(contentType)}`;
    const objectUrl = new URL(
      `${endpoint.pathname.replace(/\/$/, '')}/${awsEncode(bucket)}/${key
        .split('/')
        .map(awsEncode)
        .join('/')}`,
      endpoint.origin
    );
    const now = new Date();
    const dateStamp = now.toISOString().slice(0, 10).replaceAll('-', '');
    const amzDate = `${dateStamp}T${now
      .toISOString()
      .slice(11, 19)
      .replaceAll(':', '')}Z`;
    const payloadHash = createHash('sha256').update(bytes).digest('hex');
    const canonicalHeaders = [
      `content-type:${contentType.trim()}`,
      `host:${objectUrl.host}`,
      `x-amz-content-sha256:${payloadHash}`,
      `x-amz-date:${amzDate}`,
      '',
    ].join('\n');
    const signedHeaders = 'content-type;host;x-amz-content-sha256;x-amz-date';
    const canonicalRequest = [
      'PUT',
      objectUrl.pathname,
      '',
      canonicalHeaders.trimEnd(),
      signedHeaders,
      payloadHash,
    ].join('\n');
    const scope = `${dateStamp}/${region}/s3/aws4_request`;
    const stringToSign = [
      'AWS4-HMAC-SHA256',
      amzDate,
      scope,
      sha256(canonicalRequest),
    ].join('\n');
    const signature = sign(stringToSign, signingKey(secretKey, dateStamp, region));
    const response = await this.fetcher(objectUrl, {
      method: 'PUT',
      headers: {
        Authorization: `AWS4-HMAC-SHA256 Credential=${accessKey}/${scope}, SignedHeaders=${signedHeaders}, Signature=${signature}`,
        'Content-Type': contentType,
        'x-amz-content-sha256': payloadHash,
        'x-amz-date': amzDate,
      },
      body: Buffer.from(bytes),
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) {
      const body = (await response.text()).slice(0, 1000);
      throw new Error(`S3-compatible upload failed (${response.status}): ${body}`);
    }

    const publicBase = this.env.S3_PUBLIC_BASE_URL;
    if (publicBase) {
      const base = new URL(publicBase);
      if (base.protocol !== 'https:' || base.username || base.password) {
        throw new Error('S3_PUBLIC_BASE_URL must be an HTTPS URL without embedded credentials.');
      }
      return `${base.toString().replace(/\/$/, '')}/${key
        .split('/')
        .map(awsEncode)
        .join('/')}`;
    }
    return this.presignGet(objectUrl, accessKey, secretKey, region, dateStamp, amzDate);
  }

  private endpoint(region: string): URL {
    const configured = this.env.S3_ENDPOINT_URL;
    const endpoint = new URL(configured || `https://s3.${region}.amazonaws.com`);
    const localHttp =
      endpoint.protocol === 'http:' &&
      ['localhost', '127.0.0.1', '::1'].includes(endpoint.hostname);
    if (endpoint.protocol !== 'https:' && !localHttp) {
      throw new Error('S3_ENDPOINT_URL must use HTTPS except for localhost development.');
    }
    if (endpoint.username || endpoint.password || endpoint.search || endpoint.hash) {
      throw new Error('S3_ENDPOINT_URL must not contain credentials, query, or fragment.');
    }
    return endpoint;
  }

  private presignGet(
    objectUrl: URL,
    accessKey: string,
    secretKey: string,
    region: string,
    dateStamp: string,
    amzDate: string
  ): string {
    const scope = `${dateStamp}/${region}/s3/aws4_request`;
    const params: Record<string, string> = {
      'X-Amz-Algorithm': 'AWS4-HMAC-SHA256',
      'X-Amz-Credential': `${accessKey}/${scope}`,
      'X-Amz-Date': amzDate,
      'X-Amz-Expires': '900',
      'X-Amz-SignedHeaders': 'host',
    };
    const canonicalQuery = Object.entries(params)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => `${awsEncode(key)}=${awsEncode(value)}`)
      .join('&');
    const canonicalRequest = [
      'GET',
      objectUrl.pathname,
      canonicalQuery,
      `host:${objectUrl.host}`,
      'host',
      'UNSIGNED-PAYLOAD',
    ].join('\n');
    const stringToSign = [
      'AWS4-HMAC-SHA256',
      amzDate,
      scope,
      sha256(canonicalRequest),
    ].join('\n');
    params['X-Amz-Signature'] = sign(
      stringToSign,
      signingKey(secretKey, dateStamp, region)
    );
    const query = Object.entries(params)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => `${awsEncode(key)}=${awsEncode(value)}`)
      .join('&');
    return `${objectUrl.origin}${objectUrl.pathname}?${query}`;
  }
}

function extensionFor(contentType: string): string {
  if (contentType === 'image/jpeg') return 'jpg';
  if (contentType === 'image/svg+xml') return 'svg';
  if (contentType === 'application/json') return 'json';
  return 'png';
}

function awsEncode(value: string): string {
  return encodeURIComponent(value).replace(/[!'()*]/g, (character) =>
    `%${character.charCodeAt(0).toString(16).toUpperCase()}`
  );
}

function sha256(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function hmac(key: string | Buffer, value: string): Buffer {
  return createHmac('sha256', key).update(value).digest();
}

function signingKey(secret: string, date: string, region: string): Buffer {
  const dateKey = hmac(`AWS4${secret}`, date);
  const regionKey = hmac(dateKey, region);
  const serviceKey = hmac(regionKey, 's3');
  return hmac(serviceKey, 'aws4_request');
}

function sign(value: string, key: Buffer): string {
  return createHmac('sha256', key).update(value).digest('hex');
}
