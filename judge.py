from openai import OpenAI
import dotenv
import base64
from pathlib import Path

dotenv.load_dotenv()
client = OpenAI()

def text_reward(text, bank):
    past = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(bank)) if bank else "(none)"

    decision = client.decisions.create(
        model="gpt-6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""You are judging the output of a prompt writer whose goal is to produce diverse, wild, anti-design, and anti-aesthetic image generation prompts.

Format: each prompt may contain a '%%' separator. Text before '%%' is the positive prompt and text after '%%' is the negative prompt. If there is no '%%', the whole text is the positive prompt.

Current prompt:
{text}

Past prompts:
{past}""",
                    },
                ],
            }
        ],
        questions=[
            {
                "type": "predicate",
                "name": "validity",
                "instructions": """Answer true only if all of the following hold for the current prompt: (1) the positive prompt and the negative prompt (if present) are each readable prompts; (2) neither leaks formatting labels or meta text into the prompt itself, such as 'positive_prompt', 'negative prompt:', 'Prompt:', or explanations addressed to the user; (3) '%%' appears at most once. """,
            },
            {
                "type": "score",
                "name": "diversity",
                "instructions": """Compare the current prompt (positive and negative together) against the past prompts. Judge only the design and style choices: visual style, medium, composition, color treatment, typography, rendering technique, and which aesthetic rules are being broken. Ignore the subject matter. Two prompts about different subjects with the same style count as similar; two prompts about the same subject with very different styles count as different.""",
                "levels": [
                    {
                        "label": "homogenized",
                        "description": """The same or a closely similar style and design approach already appears several times in the past prompts.""",
                    },
                    {
                        "label": "normal",
                        "description": """The style overlaps with some past prompts but adds a noticeable new element or combination.""",
                    },
                    {
                        "label": "novel",
                        "description": """The style and design approach is clearly different from every past prompt.""",
                    },
                ],
            },
            {
                "type": "score",
                "name": "anti_design",
                "instructions": """Imagine the image a typical text-to-image model would produce from this prompt. How strongly would it break conventional design and aesthetic norms. Consider both the positive prompt and what the negative prompt pushes away from. Judge the likely image, not how edgy the wording sounds.""",
                "levels": [
                    {
                        "label": "conventional",
                        "description": """The image would look polished, pleasant, or professionally designed. Default model aesthetics dominate.""",
                    },
                    {
                        "label": "mild",
                        "description": """The image would break one or two norms (e.g., an odd color clash or awkward layout) but still read as a normal, attractive image overall.""",
                    },
                    {
                        "label": "strong",
                        "description": """The image would deliberately violate several norms and look ugly, chaotic, or uncomfortable, while still being intentional.""",
                    },
                    {
                        "label": "extreme",
                        "description": """The image would reject conventional aesthetics almost entirely in a way that feels deliberate and committed, not just broken or noisy.""",
                    },
                ],
            },
        ],
    )
    return decision.answers[0].probability * (decision.answers[1].score/2 + decision.answers[2].score/3) * 0.5 # range: 0-1


import os
index = 0
os.makedirs("images", exist_ok=True)
def image_reward(image_path, product):
  global index
  image_base64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
  decision = client.decisions.create(
      model="gpt-6-luna",
      input=[
          {
              "role": "user",
              "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_base64}",
                },
              ],
          }
      ],
      questions=[
          {
              "type": "score",
              "name": "anti_design",
              "instructions": """How strongly does this poster use anti-design? Anti-design has to reject conventional design norms intentionally, not just cool or new design.""",
              "levels": [
                  {
                      "label": "conventional",
                      "description": """The poster follows traditional design rules and looks polished or professionally designed.""",
                  },
                  {
                      "label": "mild",
                      "description": """The poster breaks one or two rules on purpose but still reads as a conventional poster overall.""",
                  },
                  {
                      "label": "strong",
                      "description": """The poster deliberately breaks several rules while the rule-breaking clearly feels intentional.""",
                  },
                  {
                      "label": "extreme",
                      "description": """The poster rejects traditional design almost entirely, and the whole piece is built around that rejection in a committed, deliberate way.""",
                  },
              ],
          },
          {
              "type": "predicate",
              "name": "valid_design",
              "instructions": (
                  f"Is this poster a valid poster for {product}? No matter the design. "
              ),
          },
      ],
  )
  return decision.answers[0].score / 3, decision.answers[1].probability # anti_design (0-1), valid_design gate (0-1)


import requests

session = requests.Session()
session.trust_env = False  # ignore http_proxy, same as wget --no-proxy

def neg_hpsv3_reward(image_path, product):
    r = session.get(
        "http://127.0.0.1:5000/",
        params={"path": image_path, "prompt": "a beautifully designed poster"},  # no product: only penalize polish, not product alignment
        timeout=300,
    )
    r.raise_for_status()
    return 1 - r.json()[0]/15


def poster_text_reward(quoted_texts, product):
    texts = "\n".join(f"- {t}" for t in quoted_texts)
    decision = client.decisions.create(
        model="gpt-6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""Product being advertised: {product}

Text that will be printed on the poster:
{texts}""",
                    },
                ],
            }
        ],
        questions=[
            {
                "type": "predicate",
                "name": "fitting_copy",
                "instructions": """Answer true only if the poster text reads as real advertising copy written for this specific product: it makes sense as words (not gibberish or random tokens), and it relates to the product, what it does, or why someone would want it. Wild, absurd, or provocative copy is fine as long as it is clearly about this product. Answer false for generic slogans that could sit on any product's poster, copy that is about a different product, or text that is just a style description rather than words meant to be printed.""",
            },
        ],
    )
    return decision.answers[0].probability # range: 0-1
