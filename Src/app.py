import gradio as gr
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

def normalize_lighting(img):
   
    img = np.array(img)

  
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    L, A, B = cv2.split(lab)

 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)

    lab2 = cv2.merge((L2, A, B))
    img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    return img2
    

model = YOLO("best.pt")


wash_guide = {
    "clean": {
        "icon": "✨",
        "title": "Clean",
        "method": "Regular washing is enough.",
    },
    "coffee": {
        "icon": "☕",
        "title": "Coffee",
        "method": """Fabric Type
        
● Cotton, polyester, spandex, nylon, jeans, towels, sheets

● Delicates: wool/silk (gentle version)

Good for:

● Black coffee, latte, iced coffee, cold brew, coffee with milk

Always:

● Flush with cold water ASAP.

● Work from inside out.

Rule:

Coffee = tannin/pigment stain.

Steps – Regular fabrics
1. Cold rinse
2. Pre-treat with enzyme detergent
3. Wash in warmest safe water
4. Repeat if stain persists
5. Air-dry before checking

Steps – Delicate fabrics
1. Cool water + mild detergent bath
2. Squeeze gently (don't scrub)
3. Warm wash if allowed
4. Air-dry""",
        "Link": "https://www.maytag.com/blog/washers-and-dryers/how-to-remove-coffee-stains-from-clothes.html"
    },
    "wine": {
        "icon": "🍷",
        "title": "Wine",
        "method": """Fabric Type
        
● Cotton, linen, blends, tablecloths, shirts

Good for:

● Red wine, grape-based stains

Always:

● Keep stain wet with cold water.

● Blot, do not rub.

Rule:

Red wine = tannin + dye stain.

Steps
1. Blot immediately (don’t scrub)
2. Cold water dilution
3. Pre-treat with heavy-duty liquid detergent (sit 1 hour)
4. Wash in warm water
5. Do not dry if tint remains
6. Dry only when fully clean
""",
        "Link": "https://laundrysauce.com/blogs/news/how-to-remove-red-wine-stain"
    },
    "blood": {
        "icon": "🩸",
        "title": "Blood",
        "method":  """Fabric Type
        
● Best: cotton, polyester, blends, bedsheets, denim

● Careful: wool, silk (use only very gentle version of this)

Good for:

● Fresh blood, dried blood, period blood, nosebleeds, small cuts
Always:

● Start with cold water, never hot at first.

● Treat as soon as possible.

Rule:

Blood = protein stain.

→ Hot water or high heat “cooks” the protein and locks the stain into the fibers.

Steps
A. Fresh blood (still wet)
1. Cold rinse from the back
2. Initial pre-treat (hydrogen peroxide or bar soap)
3. Detergent step (enzyme detergent)
4. Wash in warm water
5. Check before drying

B. Dried blood
1. Cold pre-soak (hours or overnight)
2. Pre-treat with bar soap or stain remover
3. Wash with fabric-safe bleach
4. Ammonia mix for severe stains
5. Check stain before drying""",
        "Link": "https://www.goodhousekeeping.com/home/cleaning/a69237759/get-blood-out-of-clothes/"
    },
    "chocolate": {
        "icon": "🍫",
        "title": "Chocolate",
        "method": """Fabric Type
        
● Best: cotton, polyester, blends, kids’ clothes, bedding, towels

Good for:

● Chocolate bars, melted chocolate, cocoa powder, chocolate milk, chocolate ice
cream

Always:

● Scrape off solids before adding water.

● Use cold water first, not warm.

Rule:

Chocolate = mix of fat + sugar + pigment.

Steps
1. Scrape excess chocolate
2. Cold rinse from the back
3. Pre-treat with heavy-duty detergent or dish soap
4. Cold soak 15 minutes
5. Wash in warmest safe water
6. Use stain remover if needed
7. Air-dry and check""",
        "Link": "https://www.thespruce.com/remove-chocolate-from-clothes-1901012"
    },
    "dirt_mud": {
        "icon": "🌱",
        "title": "Dirt/Mud",
        "method": """Fabric Type
        
● Cotton, polyester, sportswear, denim, outdoor clothes

Good for:

● Dry dirt, wet mud, clay, outdoor stains

Always:

● Let mud dry completely first.

● Never wipe wet mud.

Rule:

Mud = protein + minerals → rubbing when wet pushes it deeper.

Steps
1. Let dry completely
2. Scrape or vacuum dried mud
3. Pre-treat with high-performance detergent
4. Wash (warm water if allowed)
5. Deodorize with vinegar + baking soda if needed
6. Dry only when stain & odor are gone""",
        "Link": "https://www.thespruce.com/remove-mud-stains-from-clothing-1901047"
    },
    "juice": {
        "icon": "🧃",
        "title": "Juice",
        "method": """Fabric Type
        
● Cotton, polyester, blends

Good for:

● Orange juice, tangerine, grapefruit, mixed fruit juice

Always:

● Treat quickly before sugar dries.

● Use mild acid (vinegar) + detergent.

Rule:

Fruit juice = pigment + acid + sugar.

Steps
1. Blot the spill (don’t rub)
2. Apply white vinegar (acid step)
3. Dish soap → tap with soft brush
4. Rinse with lukewarm water
5. Soak in oxygen bleach at 40°C if needed
6. Wash at 40°C
7. Check before drying
""",
        "Link": "https://tide.com/en-us/how-to-wash-clothes/how-to-remove-stains/fruit-juice-stains"
    },
    "tomato_sauce": {
        "icon": "🍅",
        "title": "Ketchup",
        "method": """Fabric Type
        
● Cotton, polyester, blends, denim

Good for:

● Pasta sauce, pizza sauce, ketchup, tomato soup

Always:

● Scrape solids first.

● Rinse with cold water from back.

Rule:

Tomato = lycopene pigment + acid + often oil.

Steps – Fresh stain
1. Remove solids
2. Cold back-rinse
3. Dish soap or detergent pre-treat
4. Rinse
5. Warm wash with oxygen bleach if needed

Steps – Dried or stubborn stain
1. Scrape + cold soak
2. Use one stronger option:
○ Diluted vinegar
○ Lemon juice
○ Baking soda paste
○ Hydrogen peroxide (whites only)
○ Oxygen bleach soak
3. Wash
4. Check before drying""",
        "Link": "https://lacolada-lavanderia-autoservicio-ponferrada.com/en/blog/remove-tomato-stains-clothes/"
    },
    "ink": {
        "icon": "🖊️",
        "title": "Ink",
        "method": """Applicable Fiber Material: Cotton/Poly Blended
        
Supplies: Liquid detergent, oxygen bleach, soft brush


1. Wash the ink off your clothes with lukewarm water

2. Add 20ml of liquid detergent and 20g of oxygen bleach to 100ml of lukewarm water, stir well, and apply evenly to the contaminated area. (Colored clothes can be bleached by bleach.)

3. Rub it like you're wiping it off with a soft brush, and rinse it in cold water after 20 minutes. (Be careful because brushing strongly that the stain does not come off will damage the fabric.)

4. Add detergent and oxygen bleach together and wash them in a standard course with a water temperature of 60 degrees Celsius. (Even colored clothes can be more effective in removing stains if you put oxygen bleach together. But be careful because the color can fade if you soak it for a long time.)""",
        "Link": "https://www.lge.co.kr/story/life/thinq-discover-clothing-042#"
    },
}

def predict(img):
    if img is None:
        return "이미지를 업로드해주세요."
    
  
    img_fixed = normalize_lighting(img)
    result = model(img)[0]
    cls = result.names[result.probs.top1]
    conf = float(result.probs.top1conf)
    
    guide_info = wash_guide.get(cls, {
        "icon": "❓",
        "title": "알 수 없는 얼룩",
        "method": "세탁 정보가 등록되지 않은 얼룩입니다.",
        "tips": "전문 세탁소에 문의하세요."
    })
    

    confidence_msg = ""
    if conf >= 0.9:
        confidence_msg = "✅ Very High Confidence"
    elif conf >= 0.7:
        confidence_msg = "✓ High Confidence"
    elif conf >= 0.5:
        confidence_msg = "⚠️ Medium Confidence - Please refer to the results"
    else:
        confidence_msg = "❗ Low Confidence - Recommend re-shooting from a different angle"
    

    text = f"""
# {guide_info['icon']} {guide_info['title']}

---

## 📊 Analysis Result
- **Accuracy:** `{conf*100:.1f}%`
- **Reliability:** {confidence_msg}

---

## 🧼 How to wash

{guide_info['method']}

### Link
{guide_info['Link']}

---

### ⚠️ Precautions
- Recommended to use a professional laundry for expensive clothes or special materials
- Check the washing mark on the clothing label before washing
"""
    
    return text


custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 0.5em;
}

#description {
    text-align: center;
    font-size: 1.1em;
    color: #666;
    margin-bottom: 2em;
}

.gradio-container {
    max-width: 1000px !important;
}

#output-markdown {
    border-left: 4px solid #667eea;
    padding-left: 20px;
    background: #f8f9ff;
    border-radius: 8px;
}

#output-markdown h1:first-child {
    margin-top: 20px;
}


body, .gradio-container {
    margin: 0 auto !important;
}

.gradio-container {
    margin-left: auto !important;
    margin-right: auto !important;
}

/* 다크모드 대응 */
.dark #output-markdown {
    background: #1a1a2e !important;
    border-left-color: #667eea;
    color: #e0e0e0 !important;
}

.dark #description {
    color: #999 !important;
}

.dark #output-markdown h1,
.dark #output-markdown h2,
.dark #output-markdown h3 {
    color: #e0e0e0 !important;
}

.dark #output-markdown code {
    background: #2d2d44 !important;
    color: #a0a0ff !important;
}

.dark #output-markdown hr {
    border-color: #444 !important;
}
"""


with gr.Blocks() as demo:
    gr.HTML(f"<style>{custom_css}</style>")
    
    gr.HTML("<h1 id='title'>StainAway</h1>")
    gr.HTML("<p id='description'>Snap.Spot.Solve.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="📸 Upload a stain photo",
                height=400
            )
            
            with gr.Row():
                clear_btn = gr.ClearButton(
                    components=[input_image],
                    value="🔄 Reset"
                )
                submit_btn = gr.Button(
                    "🔍 Analyze",
                    variant="primary",
                    scale=2
                )
        
        with gr.Column(scale=1):
            output_text = gr.Markdown(
                label="Analysis results",
                elem_id="output-markdown"
            )
    

    with gr.Accordion("💁 Instructions", open=False):
        gr.Markdown("""
        1. **Photo Tips**
           - Take a picture of the stain clearly
           - It's more accurate if you shoot under bright lights
           - Take a close-up shot of the spot
        
        2. **Analyze**
           - Upload your photo and click the 'Analyze' button
           - AI identifies the type of stain and guides you on how to wash it
        
        3. **Utilize results**
           - Wash according to the suggested washing method
           - If it's not reliable, try another angle
        """)
    

    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
