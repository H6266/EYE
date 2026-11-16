#!/usr/bin/env python3
"""
AUTEUR STUDIO — IMAGE GENERATION BENCHMARK FRAMEWORK
======================================================

Shared framework for benchmarking lightweight open-weight image generation models
against SDXS-512 baseline. Designed to run on Kaggle GPU notebooks (T4/P100 16GB VRAM).

Provides:
- BENCHMARK_PROMPTS: 33 prompts with expected subjects, counts, colors, camera, lighting, environment
- GenerationResult / ReviewRecord dataclasses
- Real prompt-fidelity evaluation (human + optional VLM), NOT fabricated scores
- Contact sheet generation
- Human evaluation CSV + HTML review page
- Anonymous model labels for blinded comparison
- Results save/load
"""

import json
import time
import os
import csv
import html
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# BENCHMARK PROMPTS (33 prompts, 20 categories)
# Each prompt includes expected subjects, object counts, colors, camera angle,
# lighting, environment — used for structured prompt-fidelity review.
# =============================================================================

BENCHMARK_PROMPTS = [
    # 1-5: Detailed human portraits
    {
        "id": "portrait_001",
        "category": "human_portrait",
        "prompt": "A young African woman, approximately 30 years old, wearing a long burgundy coat and holding a transparent umbrella, standing on an old European railway station platform during heavy rain, vintage train approaching behind her, ground-level camera angle, dramatic blue-hour lighting, wet platform reflections, photorealistic, cinematic 35mm film",
        "expected_subjects": ["woman", "burgundy coat", "transparent umbrella", "railway station", "train", "rain"],
        "expected_objects_count": {"woman": 1, "umbrella": 1, "train": 1},
        "expected_colors": ["burgundy"],
        "camera_angle": "ground level",
        "lighting": "blue hour",
        "environment": "railway station platform",
        "weather": "heavy rain",
    },
    {
        "id": "portrait_002",
        "category": "human_portrait",
        "prompt": "An elderly Japanese man with white hair and a kind expression, wearing a traditional indigo-dyed kimono, sitting on a tatami mat in a tea room, natural morning light through paper shoji screens, shallow depth of field, photorealistic portrait, 85mm lens",
        "expected_subjects": ["elderly man", "white hair", "kimono", "tatami mat", "tea room"],
        "expected_objects_count": {"man": 1, "kimono": 1},
        "expected_colors": ["indigo", "white"],
        "camera_angle": "eye level",
        "lighting": "natural morning light",
        "environment": "tea room",
        "weather": "none",
    },
    {
        "id": "portrait_003",
        "category": "human_portrait",
        "prompt": "A smiling Brazilian woman with curly dark hair, wearing a bright yellow floral dress and red sandals, standing on a Copacabana beach at sunset, ocean waves behind her, golden hour lighting, photorealistic, vibrant colors",
        "expected_subjects": ["Brazilian woman", "curly dark hair", "yellow floral dress", "red sandals", "beach", "ocean waves"],
        "expected_objects_count": {"woman": 1, "dress": 1},
        "expected_colors": ["yellow", "red"],
        "camera_angle": "eye level",
        "lighting": "golden hour",
        "environment": "Copacabana beach",
        "weather": "sunny",
    },
    {
        "id": "portrait_004",
        "category": "human_portrait",
        "prompt": "A Middle Eastern man with a well-groomed beard, wearing a dark blue thobe and white ghutra, standing in a traditional souk market, spice stalls with colorful piles of spices, warm lantern lighting, photorealistic, candid expression",
        "expected_subjects": ["Middle Eastern man", "beard", "dark blue thobe", "white ghutra", "souk market", "spice stalls"],
        "expected_objects_count": {"man": 1, "thobe": 1},
        "expected_colors": ["dark blue", "white"],
        "camera_angle": "eye level",
        "lighting": "warm lantern",
        "environment": "souk market",
        "weather": "none",
    },
    {
        "id": "portrait_005",
        "category": "human_portrait",
        "prompt": "A Nordic woman with platinum blonde hair in a braid, wearing a cream-colored knit sweater and jeans, standing in a snowy forest with pine trees covered in snow, soft overcast daylight, photorealistic, serene expression",
        "expected_subjects": ["Nordic woman", "platinum blonde hair", "cream knit sweater", "jeans", "snowy forest", "pine trees"],
        "expected_objects_count": {"woman": 1, "sweater": 1},
        "expected_colors": ["platinum blonde", "cream"],
        "camera_angle": "eye level",
        "lighting": "soft overcast daylight",
        "environment": "snowy forest",
        "weather": "snowy",
    },

    # 6-10: Complex clothing and specific colors
    {
        "id": "clothing_006",
        "category": "complex_clothing",
        "prompt": "A Victorian gentleman wearing a navy blue tailcoat with gold buttons, white cravat, black top hat, and patent leather shoes, holding a silver pocket watch, standing in a foggy London street at night, gas lamp lighting",
        "expected_subjects": ["Victorian gentleman", "navy blue tailcoat", "gold buttons", "white cravat", "black top hat", "patent leather shoes", "silver pocket watch"],
        "expected_objects_count": {"gentleman": 1, "top hat": 1},
        "expected_colors": ["navy blue", "gold", "white", "black", "silver"],
        "camera_angle": "eye level",
        "lighting": "gas lamp night",
        "environment": "London street",
        "weather": "foggy",
    },
    {
        "id": "clothing_007",
        "category": "complex_clothing",
        "prompt": "A ballerina wearing a white tutu with pink satin ribbons and pointe shoes, mid-pirouette in a dance studio with mirrors and barres, natural studio lighting, motion blur on skirt, photorealistic",
        "expected_subjects": ["ballerina", "white tutu", "pink satin ribbons", "pointe shoes", "dance studio", "mirrors", "barres"],
        "expected_objects_count": {"ballerina": 1, "tutu": 1},
        "expected_colors": ["white", "pink"],
        "camera_angle": "side profile",
        "lighting": "natural studio lighting",
        "environment": "dance studio",
        "weather": "none",
    },
    {
        "id": "clothing_008",
        "category": "complex_clothing",
        "prompt": "A chef wearing a traditional white double-breasted chef jacket with black buttons, black-and-white checkered pants, tall white toque hat, standing in a professional commercial kitchen with stainless steel appliances, bright kitchen lighting",
        "expected_subjects": ["chef", "white chef jacket", "black buttons", "checkered pants", "white toque hat", "commercial kitchen", "stainless steel"],
        "expected_objects_count": {"chef": 1, "chef jacket": 1, "toque hat": 1},
        "expected_colors": ["white", "black"],
        "camera_angle": "medium shot",
        "lighting": "bright kitchen lighting",
        "environment": "commercial kitchen",
        "weather": "none",
    },
    {
        "id": "clothing_009",
        "category": "complex_clothing",
        "prompt": "A construction worker wearing an orange high-visibility vest with reflective silver stripes, yellow hard hat, work boots, holding a clipboard, standing at a construction site with scaffolding and concrete, overcast daylight",
        "expected_subjects": ["construction worker", "orange high-visibility vest", "reflective silver stripes", "yellow hard hat", "work boots", "clipboard", "scaffolding", "concrete"],
        "expected_objects_count": {"worker": 1, "vest": 1, "hard hat": 1},
        "expected_colors": ["orange", "silver", "yellow"],
        "camera_angle": "medium shot",
        "lighting": "overcast daylight",
        "environment": "construction site",
        "weather": "overcast",
    },
    {
        "id": "clothing_010",
        "category": "complex_clothing",
        "prompt": "A bride wearing an ivory lace wedding gown with a long train and veil, holding a bouquet of white roses, standing in a garden with roses and greenery, soft natural daylight, romantic atmosphere",
        "expected_subjects": ["bride", "ivory lace wedding gown", "veil", "bouquet of white roses", "garden", "roses"],
        "expected_objects_count": {"bride": 1, "wedding gown": 1},
        "expected_colors": ["ivory", "white"],
        "camera_angle": "full body",
        "lighting": "soft natural daylight",
        "environment": "garden",
        "weather": "none",
    },

    # 11-15: Animals and wildlife
    {
        "id": "animal_011",
        "category": "animal_wildlife",
        "prompt": "A red fox sitting on a moss-covered rock in an autumn forest, orange and red leaves on the ground, soft morning mist, golden sunlight filtering through trees, photorealistic wildlife photography, 300mm lens",
        "expected_subjects": ["red fox", "moss-covered rock", "autumn forest", "orange and red leaves"],
        "expected_objects_count": {"fox": 1},
        "expected_colors": ["red", "orange"],
        "camera_angle": "eye level",
        "lighting": "golden sunlight",
        "environment": "autumn forest",
        "weather": "misty",
    },
    {
        "id": "animal_012",
        "category": "animal_wildlife",
        "prompt": "A bald eagle perched on a pine branch, white head and yellow beak, blue sky background, sharp focus on feathers, wildlife photography, bright daylight",
        "expected_subjects": ["bald eagle", "white head", "yellow beak", "pine branch", "blue sky"],
        "expected_objects_count": {"eagle": 1},
        "expected_colors": ["white", "yellow", "blue"],
        "camera_angle": "side profile",
        "lighting": "bright daylight",
        "environment": "pine tree",
        "weather": "clear",
    },
    {
        "id": "animal_013",
        "category": "animal_wildlife",
        "prompt": "A sea turtle swimming underwater near a coral reef, colorful corals, small tropical fish, sunlight rays penetrating water, underwater photography, clear blue water",
        "expected_subjects": ["sea turtle", "coral reef", "colorful corals", "tropical fish", "sunlight rays"],
        "expected_objects_count": {"turtle": 1},
        "expected_colors": ["blue"],
        "camera_angle": "underwater",
        "lighting": "sunlight rays underwater",
        "environment": "coral reef",
        "weather": "none",
    },
    {
        "id": "animal_014",
        "category": "animal_wildlife",
        "prompt": "A snow leopard walking on a rocky mountain cliff, gray fur with black spots, Himalayan mountains in background, overcast mountain sky, wildlife photography, telephoto lens",
        "expected_subjects": ["snow leopard", "gray fur", "black spots", "rocky mountain cliff", "Himalayan mountains"],
        "expected_objects_count": {"snow leopard": 1},
        "expected_colors": ["gray", "black"],
        "camera_angle": "side profile",
        "lighting": "overcast mountain sky",
        "environment": "rocky mountain cliff",
        "weather": "overcast",
    },
    {
        "id": "animal_015",
        "category": "animal_wildlife",
        "prompt": "A fluffy orange tabby cat sleeping on a windowsill, sunlight streaming through the window, houseplants in background, cozy indoor atmosphere, soft warm lighting, macro photography",
        "expected_subjects": ["orange tabby cat", "windowsill", "sunlight", "houseplants"],
        "expected_objects_count": {"cat": 1},
        "expected_colors": ["orange"],
        "camera_angle": "close-up",
        "lighting": "soft warm lighting",
        "environment": "indoor windowsill",
        "weather": "none",
    },

    # 16-20: Architectural scenes and landscapes
    {
        "id": "architecture_016",
        "category": "architecture",
        "prompt": "A Gothic cathedral with flying buttresses, stained glass windows, stone spires, standing in a city square with cobblestones, dramatic sunset sky, architectural photography, wide angle",
        "expected_subjects": ["Gothic cathedral", "flying buttresses", "stained glass windows", "stone spires", "city square", "cobblestones"],
        "expected_objects_count": {"cathedral": 1},
        "expected_colors": [],
        "camera_angle": "wide angle",
        "lighting": "dramatic sunset",
        "environment": "city square",
        "weather": "clear",
    },
    {
        "id": "architecture_017",
        "category": "architecture",
        "prompt": "A modern glass skyscraper reflecting blue sky and clouds, located in a financial district, other modern buildings around, bright midday sun, architectural photography, low angle looking up",
        "expected_subjects": ["glass skyscraper", "financial district", "modern buildings"],
        "expected_objects_count": {"skyscraper": 1},
        "expected_colors": ["blue"],
        "camera_angle": "low angle",
        "lighting": "bright midday sun",
        "environment": "financial district",
        "weather": "clear",
    },
    {
        "id": "landscape_018",
        "category": "landscape",
        "prompt": "A mountain lake at dawn, mist rising from the water, pine forest on the shoreline, snow-capped mountains in the distance, pastel sky colors, landscape photography, wide angle",
        "expected_subjects": ["mountain lake", "mist", "pine forest", "snow-capped mountains", "pastel sky"],
        "expected_objects_count": {"lake": 1, "mountain": 1},
        "expected_colors": [],
        "camera_angle": "wide angle",
        "lighting": "dawn",
        "environment": "mountain lake",
        "weather": "misty",
    },
    {
        "id": "landscape_019",
        "category": "landscape",
        "prompt": "A desert canyon with red rock formations, sagebrush in the foreground, bright blue sky, harsh midday shadows, landscape photography, wide angle lens",
        "expected_subjects": ["desert canyon", "red rock formations", "sagebrush", "blue sky"],
        "expected_objects_count": {"canyon": 1},
        "expected_colors": ["red", "blue"],
        "camera_angle": "wide angle",
        "lighting": "harsh midday shadows",
        "environment": "desert canyon",
        "weather": "clear",
    },
    {
        "id": "landscape_020",
        "category": "landscape",
        "prompt": "A tropical beach with white sand, turquoise water, palm trees leaning over the water, white clouds in blue sky, bright sunny day, landscape photography",
        "expected_subjects": ["tropical beach", "white sand", "turquoise water", "palm trees", "white clouds"],
        "expected_objects_count": {"beach": 1, "palm tree": 1},
        "expected_colors": ["white", "turquoise"],
        "camera_angle": "wide angle",
        "lighting": "bright sunny day",
        "environment": "tropical beach",
        "weather": "sunny",
    },

    # 21-25: Multiple people, specific object counts and positions
    {
        "id": "multi_021",
        "category": "multiple_people",
        "prompt": "Three children playing soccer on a grass field, one child in a red jersey kicking the ball, another in a blue jersey running toward the ball, a third in yellow standing nearby, sunny afternoon, sports photography",
        "expected_subjects": ["three children", "soccer ball", "red jersey", "blue jersey", "yellow jersey", "grass field"],
        "expected_objects_count": {"children": 3, "soccer ball": 1},
        "expected_colors": ["red", "blue", "yellow"],
        "camera_angle": "wide angle",
        "lighting": "sunny afternoon",
        "environment": "grass field",
        "weather": "sunny",
    },
    {
        "id": "multi_022",
        "category": "specific_object_count",
        "prompt": "A dining table with exactly four wine glasses, three plates of food, two bottles of wine, and one candle in a glass holder, elegant dinner setting, warm candlelight, macro photography",
        "expected_objects_count": {"wine glass": 4, "plate": 3, "bottle": 2, "candle": 1},
        "expected_subjects": ["dining table", "wine glasses", "plates", "wine bottles", "candle"],
        "expected_colors": [],
        "camera_angle": "close-up",
        "lighting": "warm candlelight",
        "environment": "dinner setting",
        "weather": "none",
    },
    {
        "id": "multi_023",
        "category": "specific_object_position",
        "prompt": "A bookshelf with a red book on the top shelf, a blue book on the middle shelf, and a green book on the bottom shelf, a small plant on the left side, warm indoor lighting, shallow depth of field",
        "expected_objects_count": {"red book": 1, "blue book": 1, "green book": 1, "plant": 1},
        "expected_subjects": ["bookshelf", "red book", "blue book", "green book", "plant"],
        "expected_colors": ["red", "blue", "green"],
        "camera_angle": "close-up",
        "lighting": "warm indoor lighting",
        "environment": "indoor",
        "weather": "none",
    },
    {
        "id": "multi_024",
        "category": "complex_camera_angle",
        "prompt": "A low-angle shot looking up at a hot air balloon floating in a pink and orange sunset sky, the basket visible below the balloon, wispy clouds, aerial photography perspective",
        "expected_subjects": ["hot air balloon", "basket", "sunset sky", "clouds"],
        "expected_objects_count": {"hot air balloon": 1},
        "expected_colors": ["pink", "orange"],
        "camera_angle": "low angle looking up",
        "lighting": "sunset",
        "environment": "sky",
        "weather": "clear",
    },
    {
        "id": "multi_025",
        "category": "atmospheric_conditions",
        "prompt": "A lighthouse on a rocky cliff during a storm, dark storm clouds, rough ocean waves crashing against rocks, rain falling, dramatic lightning in the distance, moody cinematography",
        "expected_subjects": ["lighthouse", "rocky cliff", "storm clouds", "ocean waves", "rain", "lightning"],
        "expected_objects_count": {"lighthouse": 1},
        "expected_colors": [],
        "camera_angle": "wide angle",
        "lighting": "dramatic lightning",
        "environment": "rocky cliff coast",
        "weather": "stormy",
    },

    # 26-30: Styles (anime, cartoon, 3D, watercolor, fantasy)
    {
        "id": "style_026",
        "category": "anime",
        "prompt": "Anime style: A young girl with silver hair and purple eyes, wearing a school uniform with a red bow, standing under cherry blossom trees with pink petals falling, soft pastel colors, cinematic anime lighting",
        "expected_subjects": ["young girl", "silver hair", "purple eyes", "school uniform", "red bow", "cherry blossom trees"],
        "expected_objects_count": {"girl": 1},
        "expected_colors": ["silver", "purple", "red", "pink"],
        "camera_angle": "medium shot",
        "lighting": "cinematic anime lighting",
        "environment": "cherry blossom garden",
        "weather": "none",
    },
    {
        "id": "style_027",
        "category": "cartoon_2d",
        "prompt": "2D cartoon style: A cheerful orange cat wearing a detective hat and holding a magnifying glass, standing in a colorful city street with cartoon buildings, bold outlines, bright flat colors",
        "expected_subjects": ["orange cat", "detective hat", "magnifying glass", "city street", "cartoon buildings"],
        "expected_objects_count": {"cat": 1, "detective hat": 1},
        "expected_colors": ["orange"],
        "camera_angle": "medium shot",
        "lighting": "bright flat colors",
        "environment": "cartoon city street",
        "weather": "none",
    },
    {
        "id": "style_028",
        "category": "3d_animation",
        "prompt": "3D animation style: A cute robot with a round blue body and expressive eyes, holding a small potted plant, standing in a futuristic laboratory with holographic screens, Pixar-style rendering, soft global illumination",
        "expected_subjects": ["robot", "round blue body", "expressive eyes", "potted plant", "futuristic laboratory", "holographic screens"],
        "expected_objects_count": {"robot": 1, "potted plant": 1},
        "expected_colors": ["blue"],
        "camera_angle": "medium shot",
        "lighting": "soft global illumination",
        "environment": "futuristic laboratory",
        "weather": "none",
    },
    {
        "id": "style_029",
        "category": "watercolor",
        "prompt": "Watercolor painting: A Venetian canal with gondolas, historic buildings with weathered facades, reflections in the water, soft pastel colors, delicate brushwork, romantic atmosphere",
        "expected_subjects": ["Venetian canal", "gondolas", "historic buildings", "reflections"],
        "expected_objects_count": {"canal": 1, "gondola": 1},
        "expected_colors": [],
        "camera_angle": "eye level",
        "lighting": "soft pastel",
        "environment": "Venetian canal",
        "weather": "none",
    },
    {
        "id": "style_030",
        "category": "fantasy",
        "prompt": "Fantasy scene: A dragon with iridescent scales sleeping on a pile of gold coins and jewels, inside a dark cave with crystals glowing on the walls, dramatic cave lighting, epic fantasy art",
        "expected_subjects": ["dragon", "iridescent scales", "gold coins", "jewels", "dark cave", "crystals"],
        "expected_objects_count": {"dragon": 1},
        "expected_colors": [],
        "camera_angle": "wide angle",
        "lighting": "dramatic cave lighting",
        "environment": "dark cave",
        "weather": "none",
    },

    # 31-33: Complex multi-requirement prompts
    {
        "id": "complex_031",
        "category": "complex_multi_requirement",
        "prompt": "A professional chef with tattoos on both arms, wearing a white chef coat and black apron, plating a gourmet dish with microgreens and edible flowers on a white plate, in a Michelin-starred restaurant kitchen with copper pots hanging, warm kitchen lighting, photorealistic, shallow depth of field, 50mm lens",
        "expected_subjects": ["chef", "tattoos on arms", "white chef coat", "black apron", "gourmet dish", "microgreens", "edible flowers", "white plate", "Michelin-starred kitchen", "copper pots"],
        "expected_objects_count": {"chef": 1, "chef coat": 1, "plate": 1},
        "expected_colors": ["white", "black"],
        "camera_angle": "medium shot",
        "lighting": "warm kitchen lighting",
        "environment": "restaurant kitchen",
        "weather": "none",
    },
    {
        "id": "complex_032",
        "category": "complex_multi_requirement",
        "prompt": "A steampunk airship with brass fittings and large propellers, floating above a Victorian city with clock towers and gas lamps, smoke coming from the airship's engine, sunset sky with orange and purple clouds, cinematic wide shot, photorealistic",
        "expected_subjects": ["steampunk airship", "brass fittings", "propellers", "Victorian city", "clock towers", "gas lamps", "smoke", "sunset sky"],
        "expected_objects_count": {"airship": 1},
        "expected_colors": ["brass", "orange", "purple"],
        "camera_angle": "cinematic wide shot",
        "lighting": "sunset",
        "environment": "sky above Victorian city",
        "weather": "clear",
    },
    {
        "id": "complex_033",
        "category": "complex_multi_requirement",
        "prompt": "An elderly woman with short gray hair, wearing reading glasses and a floral dress, knitting a blue scarf, sitting in a rocking chair in a sunlit living room with bookshelves and a fireplace, warm afternoon light, photorealistic, intimate portrait",
        "expected_subjects": ["elderly woman", "short gray hair", "reading glasses", "floral dress", "blue scarf", "rocking chair", "living room", "bookshelves", "fireplace"],
        "expected_objects_count": {"woman": 1, "scarf": 1, "rocking chair": 1},
        "expected_colors": ["gray", "blue"],
        "camera_angle": "medium shot",
        "lighting": "warm afternoon light",
        "environment": "sunlit living room",
        "weather": "none",
    },
]

PROMPT_LOOKUP = {p["id"]: p for p in BENCHMARK_PROMPTS}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GenerationResult:
    model_id: str
    prompt_id: str
    category: str = ""
    image_path: str = ""
    generation_time_s: float = 0.0
    vram_peak_gb: Optional[float] = None
    ram_peak_gb: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] | None = None


@dataclass
class ReviewRecord:
    """
    Structured review record for ONE generated image.
    Filled by human evaluator (or optional VLM). Scores are NOT auto-fabricated.
    """
    model_id: str
    prompt_id: str
    prompt: str
    image_path: str
    expected_subjects: List[str] = field(default_factory=list)
    expected_objects_count: Dict[str, int] = field(default_factory=dict)
    expected_colors: List[str] = field(default_factory=list)
    camera_angle: str = ""
    lighting: str = ""
    environment: str = ""
    generation_time_s: float = 0.0
    vram_peak_gb: Optional[float] = None

    # Human evaluation fields (filled by reviewer, NOT auto-filled)
    subject_presence_score: Optional[float] = None      # 0-1, all requested subjects present?
    object_count_accuracy: Optional[float] = None       # 0-1, correct object counts?
    color_accuracy: Optional[float] = None              # 0-1, specified colors correct?
    composition_score: Optional[float] = None           # 0-1, objects in requested positions?
    lighting_accuracy: Optional[float] = None           # 0-1, lighting matches?
    environment_match: Optional[float] = None           # 0-1, environment correct?
    unwanted_elements: bool = False                     # hallucinated objects/people?
    overall_accuracy: Optional[float] = None            # 0-1 holistic
    human_notes: str = ""
    missing_details: List[str] = field(default_factory=list)
    incorrect_details: List[str] = field(default_factory=list)
    unwanted_additions: List[str] = field(default_factory=list)

    # VLM evaluation (optional, NOT ground truth)
    vlm_subject_presence: Optional[float] = None
    vlm_object_count: Optional[float] = None
    vlm_color_accuracy: Optional[float] = None
    vlm_overall: Optional[float] = None
    vlm_notes: str = ""
    vlm_evaluated: bool = False


@dataclass
class AnonymousLabel:
    """Maps real model ID to anonymous label for blinded human evaluation."""
    model_id: str
    anonymous_label: str  # e.g. "Model A", "Model B"


# =============================================================================
# EVALUATION SYSTEM
# =============================================================================

EVALUATION_CRITERIA = {
    "subject_presence": "Were all requested subjects included?",
    "object_count": "Were the correct number of objects generated?",
    "color_accuracy": "Were requested colors correct?",
    "composition": "Were objects positioned correctly?",
    "lighting": "Was the requested lighting correct?",
    "environment": "Was the intended environment present?",
    "detail_preservation": "Were important details not omitted?",
    "no_unwanted": "Did the model avoid inventing unwanted elements?",
}


def evaluate_generation(
    model_id: str,
    prompt_entry: Dict[str, Any],
    image_path: str,
    vlm_evaluator=None,
    human_score: Optional[float] = None,
) -> ReviewRecord:
    """
    Build a ReviewRecord for a generated image.

    Scores are NOT auto-fabricated. All numeric fields default to None and must
    be filled by a human evaluator (or optional VLM, clearly marked as such).
    """
    return ReviewRecord(
        model_id=model_id,
        prompt_id=prompt_entry["id"],
        prompt=prompt_entry["prompt"],
        image_path=image_path,
        expected_subjects=prompt_entry.get("expected_subjects", []),
        expected_objects_count=prompt_entry.get("expected_objects_count", {}),
        expected_colors=prompt_entry.get("expected_colors", []),
        camera_angle=prompt_entry.get("camera_angle", ""),
        lighting=prompt_entry.get("lighting", ""),
        environment=prompt_entry.get("environment", ""),
        human_notes="PENDING HUMAN EVALUATION",
    )


def review_record_to_csv_row(record: ReviewRecord, anonymous_label: Optional[str] = None) -> Dict[str, Any]:
    """Flatten a ReviewRecord into a CSV row."""
    model_label = anonymous_label if anonymous_label else record.model_id
    return {
        "anonymous_label": model_label,
        "model_id": record.model_id,
        "prompt_id": record.prompt_id,
        "prompt": record.prompt,
        "image_path": record.image_path,
        "expected_subjects": ";".join(record.expected_subjects),
        "expected_objects_count": json.dumps(record.expected_objects_count),
        "expected_colors": ";".join(record.expected_colors),
        "camera_angle": record.camera_angle,
        "lighting": record.lighting,
        "environment": record.environment,
        "generation_time_s": record.generation_time_s,
        "vram_peak_gb": record.vram_peak_gb,
        "subject_presence_score": record.subject_presence_score,
        "object_count_accuracy": record.object_count_accuracy,
        "color_accuracy": record.color_accuracy,
        "composition_score": record.composition_score,
        "lighting_accuracy": record.lighting_accuracy,
        "environment_match": record.environment_match,
        "unwanted_elements": record.unwanted_elements,
        "overall_accuracy": record.overall_accuracy,
        "human_notes": record.human_notes,
        "missing_details": ";".join(record.missing_details),
        "incorrect_details": ";".join(record.incorrect_details),
        "unwanted_additions": ";".join(record.unwanted_additions),
        "vlm_subject_presence": record.vlm_subject_presence,
        "vlm_object_count": record.vlm_object_count,
        "vlm_color_accuracy": record.vlm_color_accuracy,
        "vlm_overall": record.vlm_overall,
        "vlm_evaluated": record.vlm_evaluated,
        "vlm_notes": record.vlm_notes,
    }


# =============================================================================
# CONTACT SHEET
# =============================================================================

def _load_font(size: int = 20) -> ImageFont.FreeTypeFont:
    """Try to load a truetype font, fall back to default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_contact_sheet(
    image_entries: List[Tuple[str, str]],
    output_path: str,
    cols: int = 4,
    thumb_size: Tuple[int, int] = (300, 300),
    label_font_size: int = 16,
):
    """
    Create a contact sheet showing generated images side by side with prompt IDs.

    image_entries: list of (image_path, label) tuples.
    """
    if not image_entries:
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    font = _load_font(label_font_size)
    thumb_w, thumb_h = thumb_size
    rows = (len(image_entries) + cols - 1) // cols
    sheet_w = cols * (thumb_w + 20) + 40
    sheet_h = rows * (thumb_h + 60) + 60

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    for idx, (img_path, label) in enumerate(image_entries):
        r = idx // cols
        c = idx % cols
        x = 20 + c * (thumb_w + 20)
        y = 20 + r * (thumb_h + 60)

        try:
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            # Paste centered on a black thumb-sized canvas
            canvas = Image.new("RGB", (thumb_w, thumb_h), color=(0, 0, 0))
            canvas.paste(img, ((thumb_w - img.size[0]) // 2, (thumb_h - img.size[1]) // 2))
            sheet.paste(canvas, (x, y))
        except Exception as e:
            # Draw error placeholder
            draw.rectangle([x, y, x + thumb_w, y + thumb_h], fill=(60, 0, 0))
            draw.text((x + 10, y + 10), f"LOAD ERROR", fill=(255, 255, 255), font=font)

        # Draw label below image
        label_y = y + thumb_h + 5
        draw.text((x, label_y), label, fill=(220, 220, 220), font=font)

    sheet.save(output_path)
    return output_path


# =============================================================================
# ANONYMOUS LABELS
# =============================================================================

def assign_anonymous_labels(model_ids: List[str]) -> List[AnonymousLabel]:
    """
    Assign anonymous labels (Model A, Model B, ...) to model IDs for blinded
    human comparison. Reduces selection bias.
    """
    labels = []
    for idx, mid in enumerate(sorted(set(model_ids))):
        letter = chr(ord("A") + idx)
        labels.append(AnonymousLabel(model_id=mid, anonymous_label=f"Model {letter}"))
    return labels


def get_anonymous_label(model_id: str, labels: List[AnonymousLabel]) -> str:
    for l in labels:
        if l.model_id == model_id:
            return l.anonymous_label
    return model_id


# =============================================================================
# HUMAN EVALUATION EXPORT (CSV + HTML)
# =============================================================================

def save_human_evaluation(
    review_records: List[ReviewRecord],
    csv_path: str,
    html_path: str,
    anonymous_labels: Optional[List[AnonymousLabel]] = None,
):
    """
    Save human evaluation CSV and HTML review page from review records.
    Includes all fields needed for human scoring.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)

    # CSV
    fieldnames = [
        "anonymous_label", "model_id", "prompt_id", "prompt", "image_path",
        "expected_subjects", "expected_objects_count", "expected_colors",
        "camera_angle", "lighting", "environment",
        "generation_time_s", "vram_peak_gb",
        "subject_presence_score", "object_count_accuracy", "color_accuracy",
        "composition_score", "lighting_accuracy", "environment_match",
        "unwanted_elements", "overall_accuracy", "human_notes",
        "missing_details", "incorrect_details", "unwanted_additions",
        "vlm_subject_presence", "vlm_object_count", "vlm_color_accuracy",
        "vlm_overall", "vlm_evaluated", "vlm_notes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in review_records:
            row = review_record_to_csv_row(r, get_anonymous_label(r.model_id, anonymous_labels) if anonymous_labels else None)
            writer.writerow(row)

    # HTML review page
    html_rows = []
    for r in review_records:
        anon = get_anonymous_label(r.model_id, anonymous_labels) if anonymous_labels else r.model_id
        img_tag = f'<img src="{html.escape(r.image_path)}" width="220" alt="{r.prompt_id}" style="border:1px solid #444;"/>' if r.image_path and os.path.exists(r.image_path) else '<span style="color:#ff6b6b;">IMAGE NOT GENERATED</span>'
        html_rows.append(f"""
        <tr>
            <td>{html.escape(anon)}</td>
            <td>{html.escape(r.prompt_id)}</td>
            <td>{html.escape(r.prompt[:120])}...</td>
            <td>{img_tag}</td>
            <td>{"; ".join(r.expected_subjects)}</td>
            <td>{json.dumps(r.expected_objects_count)}</td>
            <td>{"; ".join(r.expected_colors)}</td>
            <td>{html.escape(r.camera_angle)}</td>
            <td>{html.escape(r.lighting)}</td>
            <td>{html.escape(r.environment)}</td>
            <td>{r.subject_presence_score if r.subject_presence_score is not None else ""}</td>
            <td>{r.object_count_accuracy if r.object_count_accuracy is not None else ""}</td>
            <td>{r.color_accuracy if r.color_accuracy is not None else ""}</td>
            <td>{r.composition_score if r.composition_score is not None else ""}</td>
            <td>{r.lighting_accuracy if r.lighting_accuracy is not None else ""}</td>
            <td>{r.environment_match if r.environment_match is not None else ""}</td>
            <td>{"Yes" if r.unwanted_elements else "No"}</td>
            <td>{r.overall_accuracy if r.overall_accuracy is not None else ""}</td>
            <td>{html.escape(r.human_notes)}</td>
            <td>{"; ".join(r.missing_details)}</td>
            <td>{"; ".join(r.incorrect_details)}</td>
            <td>{"; ".join(r.unwanted_additions)}</td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Human Prompt-Fidelity Evaluation</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
  h1 {{ color: #ffd700; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
  th, td {{ border: 1px solid #444; padding: 8px; text-align: left; font-size: 13px; }}
  th {{ background: #333; color: #ffd700; position: sticky; top: 0; }}
  tr:nth-child(even) {{ background: #222; }}
  img {{ display: block; max-width: 100%; height: auto; }}
  .legend {{ margin: 20px 0; color: #888; font-size: 13px; }}
  .anon-label {{ font-weight: bold; color: #ffd700; }}
  .note {{ font-style: italic; color: #888; }}
</style>
</head>
<body>
<h1>Human Prompt-Fidelity Evaluation</h1>
<p class="note">Generated: {datetime.now().isoformat()} | Total records: {len(review_records)}</p>
<p class="legend">Anonymous labels (Model A, B, ...) hide real model identity to reduce selection bias.
Scores 0-1 are filled by human evaluators. VLM columns are optional automated scores, NOT ground truth.</p>
<table>
<thead>
<tr>
<th>Anonymous Label</th><th>Prompt ID</th><th>Prompt (truncated)</th><th>Image</th>
<th>Expected Subjects</th><th>Expected Object Counts</th><th>Expected Colors</th>
<th>Camera Angle</th><th>Lighting</th><th>Environment</th>
<th>Subject Presence (0-1)</th><th>Object Count (0-1)</th><th>Color Accuracy (0-1)</th>
<th>Composition (0-1)</th><th>Lighting (0-1)</th><th>Environment (0-1)</th>
<th>Unwanted Elements</th><th>Overall (0-1)</th><th>Human Notes</th>
<th>Missing Details</th><th>Incorrect Details</th><th>Unwanted Additions</th>
</tr>
</thead>
<tbody>
{''.join(html_rows)}
</tbody>
</table>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"Human evaluation CSV: {csv_path}")
    print(f"Human evaluation HTML: {html_path}")
    return csv_path, html_path


# =============================================================================
# RESULTS MANAGEMENT
# =============================================================================

def save_results(results: List[Dict], filepath: str):
    """Save benchmark results to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_prompts": len(BENCHMARK_PROMPTS),
            "results": results,
        }, f, indent=2)


def load_results(filepath: str) -> List[Dict]:
    """Load benchmark results from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return data["results"]


def save_review_records(records: List[ReviewRecord], filepath: str):
    """Save review records to JSON."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_records": len(records),
            "records": [asdict(r) for r in records],
        }, f, indent=2)


def load_review_records(filepath: str) -> List[ReviewRecord]:
    """Load review records from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return [ReviewRecord(**r) for r in data["records"]]


def generate_comparison_table(all_results: Dict[str, List[Dict]]) -> str:
    """Generate markdown comparison table from all model results."""
    lines = [
        "| Metric | " + " | ".join(all_results.keys()) + " |",
        "|" + "---|" * (len(all_results) + 1),
    ]
    metrics = [
        "Success rate", "Avg generation time (s)", "Peak VRAM (GB)", "Prompt fidelity (human)",
    ]
    for metric in metrics:
        values = " | ".join([""] * len(all_results))
        lines.append(f"| {metric} | {values} |")
    return "\n".join(lines)
