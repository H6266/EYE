import fs from 'fs';
import path from 'path';

export const CLIP_MAX_LENGTH = 77;
export const CLIP_BOS_TOKEN_ID = 49406;
export const CLIP_EOS_TOKEN_ID = 49407;

function bytesToUnicode(): Map<number, string> {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const map = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) {
    map.set(bs[i], String.fromCharCode(cs[i]));
  }
  return map;
}

function getPairs(word: string[]): Array<[string, string]> {
  const pairs: Array<[string, string]> = [];
  if (word.length < 2) return pairs;
  let prev = word[0];
  for (let i = 1; i < word.length; i++) {
    pairs.push([prev, word[i]]);
    prev = word[i];
  }
  return pairs;
}

const CLIP_TOKEN_PATTERN =
  /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;

export interface TokenizedPrompt {
  inputIds: BigInt64Array;
  activeTokenCount: number;
  maxTokens: number;
  truncated: boolean;
}

export class ClipBpeTokenizer {
  private readonly encoder: Record<string, number>;
  private readonly bpeRanks: Map<string, number>;
  private readonly byteEncoder: Map<number, string>;
  private readonly cache = new Map<string, string>();

  constructor(tokenizerDir: string) {
    const vocabRaw = fs.readFileSync(path.join(tokenizerDir, 'vocab.json'), 'utf8');
    this.encoder = JSON.parse(vocabRaw) as Record<string, number>;

    const mergesRaw = fs.readFileSync(path.join(tokenizerDir, 'merges.txt'), 'utf8');
    const mergeLines = mergesRaw.split('\n').slice(1, 49152 - 256 - 2 + 1);
    this.bpeRanks = new Map<string, number>();
    mergeLines.forEach((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      this.bpeRanks.set(trimmed, idx);
    });

    this.byteEncoder = bytesToUnicode();
  }

  private bpe(token: string): string {
    const cached = this.cache.get(token);
    if (cached !== undefined) return cached;

    if (token.length === 0) return '';
    let word = [...token.slice(0, -1), `${token.slice(-1)}</w>`];
    let pairs = getPairs(word);

    if (pairs.length === 0) {
      const out = `${token}</w>`;
      this.cache.set(token, out);
      return out;
    }

    while (true) {
      let bestPair: [string, string] | null = null;
      let bestRank = Infinity;

      for (const pair of pairs) {
        const key = `${pair[0]} ${pair[1]}`;
        const rank = this.bpeRanks.get(key);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = pair;
        }
      }

      if (!bestPair) break;

      const [first, second] = bestPair;
      const newWord: string[] = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }
        newWord.push(...word.slice(i, j));
        if (j < word.length - 1 && word[j + 1] === second) {
          newWord.push(first + second);
          i = j + 2;
        } else {
          newWord.push(word[j]);
          i = j + 1;
        }
      }

      word = newWord;
      if (word.length === 1) break;
      pairs = getPairs(word);
    }

    const joined = word.join(' ');
    this.cache.set(token, joined);
    return joined;
  }

  encode(text: string): TokenizedPrompt {
    const normalized = text.replace(/\s+/g, ' ').trim().toLowerCase();
    const matches = normalized.match(CLIP_TOKEN_PATTERN) || [];

    const bpeTokens: number[] = [];
    const utf8 = new TextEncoder();

    for (const rawToken of matches) {
      const bytes = utf8.encode(rawToken);
      let encodedStr = '';
      for (const b of bytes) {
        encodedStr += this.byteEncoder.get(b) || '';
      }
      const bpeOut = this.bpe(encodedStr);
      for (const piece of bpeOut.split(' ')) {
        if (!piece) continue;
        const id = this.encoder[piece];
        if (id !== undefined) {
          bpeTokens.push(id);
        }
      }
    }

    const maxBody = CLIP_MAX_LENGTH - 2;
    const truncated = bpeTokens.length > maxBody;
    const body = bpeTokens.slice(0, maxBody);

    const inputIds = new BigInt64Array(CLIP_MAX_LENGTH).fill(BigInt(CLIP_EOS_TOKEN_ID));
    inputIds[0] = BigInt(CLIP_BOS_TOKEN_ID);
    for (let i = 0; i < body.length; i++) {
      inputIds[i + 1] = BigInt(body[i]);
    }
    inputIds[body.length + 1] = BigInt(CLIP_EOS_TOKEN_ID);

    return {
      inputIds,
      activeTokenCount: body.length + 2,
      maxTokens: CLIP_MAX_LENGTH,
      truncated,
    };
  }
}
