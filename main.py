from paddleocr import PaddleOCR
import cv2
import os
import pandas as pd

ocr = PaddleOCR(use_doc_orientation_classify=False,
                use_doc_unwarping= False,
                enable_mkldnn=False)

folder = r"C:\Users\ln.kha\Downloads\Ins_Img"

results = []

for file in os.listdir(folder):
    if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):

        img_path = os.path.join(folder, file)

        result = ocr.predict(img_path)

        text = ""

        for page in result:
            for line in page["rec_texts"]:
                text += line + " "

        results.append({
            "file_name": file,
            "text": text.strip()
        })

df = pd.DataFrame(results)
df.to_csv("ocr_output.csv", index=False)
