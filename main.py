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
        _, thresh = cv2.threshold(
            img,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (15, 1)
        )

        closed = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            close_kernel
        )
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (img.shape[1] // 20, 1)
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, img.shape[0] // 20)
        )

        horizontal = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            horizontal_kernel
        )
        vertical = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            vertical_kernel
        )
        table_mask = cv2.add(horizontal, vertical)
        contours, hierarchy = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []

        for i, cnt in enumerate(contours):

            parent = hierarchy[0][i][3]

            if parent != -1:
                valid_contours.append(cnt)

        print("Cell contours:", len(valid_contours))
        #convert color
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)



        for i, cnt in enumerate(valid_contours):
            cv2.drawContours(
                debug_img,
                [cnt],
                -1,
                (255, 0, 0),
                2
            )

        img_h, img_w = img_color.shape[:2]
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)

            if w > img_w *0.05 and h> img_h * 0.05:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cells.append((x, y, w, h))
            cv2.putText(
                debug_img,
                str(i),
                (x,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0),
                1
            )
        cells.sort(key=lambda c: (c[1], c[0]))
        table_rows = []
        debug_folder = "debug"
        os.makedirs(debug_folder, exist_ok=True)

        for i,(x, y, w, h) in enumerate(cells[:250]):
            pad = 3

            crop = img_color[
                max(0, y + pad):min(img.shape[0], y + h - pad),
                max(0, x + pad):min(img.shape[1], x + w - pad)
            ]
            cv2.imwrite(os.path.join(debug_folder,f"{file}_crop_{i}.png"), crop)
            result = ocr.predict(crop)

            text = ""

            if result and result[0]["rec_texts"]:
                text = "".join(result[0]["rec_texts"])

            found = False
            call_center_y = y+h//2
            for row in table_rows:
                if abs(row["y"] - call_center_y) < 10:
                    row["cells"].append((x, text))
                    found = True
                    break

            if not found:
                table_rows.append({
                    "y": call_center_y,
                    "cells": [(x, text)]
                })

        final_table = []

        for row in table_rows:
            ordered = sorted(
                row["cells"],
                key=lambda v: v[0]
            )

            final_table.append(
                [f";{text}" for _, text in ordered]
            )

        table_rows.sort(key=lambda r: r["y"])

        df = pd.DataFrame(final_table)

        csv_name = os.path.splitext(file)[0] + ".csv"

        df.to_csv(
            csv_name,
            index=False,
            header=False,
            encoding="utf-8-sig"
        )



        print(f"{file}: {len(cells)} cells")








