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
        cells =[]
        img = cv2.imread(img_path, 0)
        thresh = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            5
        )
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (50, 1)
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 50)
        )

        horizontal = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            horizontal_kernel
        )

        vertical = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            vertical_kernel
        )
        table_mask = cv2.add(horizontal, vertical)
        contours, hierachy = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("Contours:", len(contours))
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if 30 < w < 300 and 20 < h < 100:
                cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cells.append((x, y, w, h))
        cells.sort(key=lambda c: (c[1], c[0]))
        table_rows = []
        debug_folder = "debug"
        os.makedirs(debug_folder, exist_ok=True)

        for i,(x, y, w, h) in enumerate(cells):

            crop = img[
                y + 2:y + h - 2,
                x + 2:x + w - 2
            ]

            if crop.size == 0:
                continue

            crop_bgr = cv2.cvtColor(
                crop,
                cv2.COLOR_GRAY2BGR
            )
            if i < 20:
                cv2.imwrite(
                    os.path.join(
                        debug_folder,
                        f"{file}_crop_{i}.png"
                    ),
                    crop
                )

            result = ocr.predict(crop_bgr)

            text = ""

            if result and result[0]["rec_texts"]:
                text = result[0]["rec_texts"][0]

            found = False

            for row in table_rows:
                if abs(row["y"] - y) < 10:
                    row["cells"].append((x, text))
                    found = True
                    break

            if not found:
                table_rows.append({
                    "y": y,
                    "cells": [(x, text)]
                })

        final_table = []

        for row in table_rows:
            ordered = sorted(
                row["cells"],
                key=lambda v: v[0]
            )

            final_table.append(
                [text for _, text in ordered]
            )

        table_rows.sort(key=lambda r: r["y"])

        df = pd.DataFrame(final_table)

        csv_name = os.path.splitext(file)[0] + ".csv"
        cv2.imwrite(
            os.path.join(debug_folder, f"{file}_cells.png"),
            img_color
        )
        cv2.imwrite(
            os.path.join(debug_folder, f"{file}_table_mask.png"),
            table_mask
        )

        df.to_csv(
            csv_name,
            index=False,
            header=False,
            encoding="utf-8-sig"
        )



        print(f"{file}: {len(cells)} cells")








