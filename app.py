# -*- coding: utf-8 -*-
"""
app.py - เว็บแอป Streamlit "AI ทำนายระดับความเสี่ยงในการลงทุน"

หมายเหตุสำคัญ:
ไฟล์ *.pkcls เป็นโมเดลที่บันทึกมาจากโปรแกรม Orange3 (ไม่ใช่โมเดล scikit-learn เพียว ๆ)
ดังนั้นต้องติดตั้งไลบรารี Orange3 ด้วย จึงจะ joblib.load ได้
โมเดลของ Orange จะ "ทำ Normalize และ One-hot encode ให้เองอัตโนมัติ"
ตามขั้นตอนที่ตั้งไว้ใน Workflow ตอนฝึก เราจึงแค่ส่งค่าดิบ (ก่อนแปลง) เข้าไปให้ถูกชื่อคอลัมน์และถูกชนิดข้อมูล
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from Orange.data import Domain, Table

# ---------------------------------------------------------------------------
# 1) ตั้งค่าหน้าเว็บและรายชื่อโมเดล
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI ทำนายระดับความเสี่ยงในการลงทุน", page_icon="📈", layout="wide")

BASE_DIR = Path(__file__).parent  # โฟลเดอร์เดียวกับไฟล์ app.py

# ชื่อที่แสดงในเมนู -> ชื่อไฟล์โมเดล (วางไฟล์ไว้โฟลเดอร์เดียวกับ app.py)
MODEL_FILES = {
    "Naive Bayes": "naive_bayes.pkcls",
    "SVM": "svm.pkcls",
    "Neural Network": "Neural_Network.pkcls",
}

# ---------------------------------------------------------------------------
# 2) ตั้งค่าช่องกรอกของตัวแปรแต่ละตัว (ชื่อต้องตรงกับชื่อคอลัมน์ตอนฝึกโมเดลเป๊ะ ๆ)
#    - label  : ข้อความที่แสดงให้ผู้ใช้เห็น
#    - ตัวเลข : min, max, default, step
#    - ข้อความ (หมวดหมู่) : ไม่ต้องกำหนด ระบบดึงตัวเลือกจากโมเดลให้เอง
# ---------------------------------------------------------------------------
NUMERIC_CONFIG = {
    "Income":               dict(label="รายได้ต่อปี",                       min=0,   max=10_000_000, default=70000,  step=1000),
    "Credit Score":         dict(label="คะแนนเครดิต",                      min=300, max=850,        default=700,    step=1),
    "Loan Amount":          dict(label="ยอดเงินกู้",                        min=0,   max=10_000_000, default=27000,  step=1000),
    "Years at Current Job": dict(label="อายุงานปัจจุบัน (ปี)",              min=0,   max=60,         default=9,      step=1),
    "Debt-to-Income Ratio": dict(label="สัดส่วนหนี้ต่อรายได้ (0-1)",        min=0.0, max=1.0,        default=0.35,   step=0.01),
    "Assets Value":         dict(label="มูลค่าสินทรัพย์",                   min=0,   max=50_000_000, default=160000, step=1000),
    "Previous Defaults":    dict(label="จำนวนครั้งที่เคยผิดนัดชำระ",        min=0,   max=10,         default=2,      step=1),
}

# ป้ายภาษาไทยของคอลัมน์หมวดหมู่ (ถ้าไม่ได้ระบุ จะใช้ชื่อคอลัมน์เดิมเป็นภาษาอังกฤษ)
CATEGORY_LABELS = {
    "Loan Purpose": "วัตถุประสงค์การกู้",
    "Employment Status": "สถานะการจ้างงาน",
    "Payment History": "ประวัติการชำระเงิน",
}

# ซ่อนช่องกรอกที่ไม่ต้องการ โดยกำหนดค่าคงที่ให้แทน (โมเดลเดิมยังต้องการคอลัมน์ครบ จึงตัดทิ้งเฉยๆ ไม่ได้)
# รูปแบบ  "ชื่อคอลัมน์": ค่าที่จะส่งเข้าโมเดลเสมอ   ตัวอย่างด้านล่างเอา # ออกเพื่อเปิดใช้
HIDDEN_FEATURES = {
    # "Employment Status": "Employed",
    # "Previous Defaults": 0,
}

# แปลผลลัพธ์ (ค่าคลาสในโมเดล) เป็นภาษาไทย
RISK_TEXT = {"Low": "ต่ำ", "Medium": "ปานกลาง", "High": "สูง"}


# ---------------------------------------------------------------------------
# 3) โหลดโมเดลด้วย joblib (cache ไว้ จะได้โหลดครั้งเดียวตอนเริ่มแอป)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(file_name: str):
    return joblib.load(BASE_DIR / file_name)


def get_raw_variables(model):
    """
    ดึงรายการตัวแปร "ดิบ" (ก่อน Normalize / One-hot) ที่โมเดลต้องการ
    ตอนฝึกใน Orange ตัวแปรตัวเลขถูก Normalize ผ่าน compute_value ซึ่งอ้างอิงตัวแปรดิบอยู่
    เราจึงย้อนกลับไปหยิบตัวแปรดิบนั้นมาใช้สร้างช่องกรอกให้ตรงกับตอนฝึก
    """
    raw_vars = []
    for var in model.original_domain.attributes:
        cv = getattr(var, "compute_value", None)
        raw_vars.append(cv.variable if cv is not None and hasattr(cv, "variable") else var)
    return raw_vars


# ---------------------------------------------------------------------------
# 4) ส่วนหัวของแอป
# ---------------------------------------------------------------------------
st.title("AI ทำนายระดับความเสี่ยงในการลงทุน")
st.caption("กรอกข้อมูลของผู้ลงทุน เลือกโมเดล แล้วกดปุ่ม \"ทำนายผล\"")

# ---------------------------------------------------------------------------
# 5) ให้ผู้ใช้เลือกโมเดลเอง
# ---------------------------------------------------------------------------
model_name = st.selectbox("เลือกโมเดลที่ต้องการใช้ทำนาย", list(MODEL_FILES.keys()))

try:
    model = load_model(MODEL_FILES[model_name])
except FileNotFoundError:
    st.error(f"ไม่พบไฟล์โมเดล {MODEL_FILES[model_name]} กรุณาวางไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

raw_vars = get_raw_variables(model)
class_values = list(model.domain.class_var.values)  # เช่น ['High', 'Low', 'Medium']

# ---------------------------------------------------------------------------
# 6) สร้างช่องกรอกข้อมูล (features) ทุกตัว
#    - คอลัมน์หมวดหมู่ -> st.selectbox   - คอลัมน์ตัวเลข -> st.number_input
# ---------------------------------------------------------------------------
st.subheader("ข้อมูลผู้ลงทุน")
inputs = {}
cols = st.columns(3)  # จัดช่องกรอกเป็น 3 คอลัมน์ให้ดูง่าย

visible_vars = [v for v in raw_vars if v.name not in HIDDEN_FEATURES]
for v in raw_vars:
    if v.name in HIDDEN_FEATURES:
        inputs[v.name] = HIDDEN_FEATURES[v.name]  # ใช้ค่าคงที่แทนการให้ผู้ใช้กรอก

for i, var in enumerate(visible_vars):
    with cols[i % 3]:
        if var.is_discrete:
            # คอลัมน์ข้อความ/หมวดหมู่ ใช้ selectbox โดยดึงตัวเลือกจากโมเดล
            inputs[var.name] = st.selectbox(
                CATEGORY_LABELS.get(var.name, var.name),
                list(var.values),
                key=f"in_{var.name}",
            )
        else:
            # คอลัมน์ตัวเลข ใช้ number_input
            cfg = NUMERIC_CONFIG.get(
                var.name, dict(label=var.name, min=0.0, max=1e9, default=0.0, step=1.0)
            )
            is_int = all(isinstance(cfg[k], int) for k in ("min", "max", "default", "step"))
            conv = int if is_int else float
            inputs[var.name] = st.number_input(
                cfg["label"],
                min_value=conv(cfg["min"]),
                max_value=conv(cfg["max"]),
                value=conv(cfg["default"]),
                step=conv(cfg["step"]),
                key=f"in_{var.name}",
            )

# ---------------------------------------------------------------------------
# 7) ปุ่มทำนายผล
# ---------------------------------------------------------------------------
if st.button("ทำนายผล", type="primary"):
    # 7.1 จัดข้อมูลเป็น DataFrame 1 แถว เรียงคอลัมน์ให้ตรงกับตอนฝึกโมเดล
    df = pd.DataFrame([[inputs[v.name] for v in raw_vars]], columns=[v.name for v in raw_vars])

    # 7.2 แปลงเป็นตาราง Orange: ข้อความ -> ดัชนีหมวดหมู่ (to_val), ตัวเลข -> float
    #     จากนั้นโมเดลจะ Normalize + One-hot encode ภายในเองตามที่ตั้งไว้ตอนฝึก
    row = [v.to_val(inputs[v.name]) if v.is_discrete else float(inputs[v.name]) for v in raw_vars]
    table = Table.from_numpy(Domain(raw_vars), np.array([row], dtype=float))

    # 7.3 ทำนายคลาส และความน่าจะเป็นของแต่ละคลาส
    pred_idx = int(model(table)[0])
    probs = model(table, model.Probs)[0]
    pred_label = class_values[pred_idx]
    pred_th = RISK_TEXT.get(pred_label, pred_label)

    # 7.4 แสดงผลให้อ่านเข้าใจง่าย (สีเขียว/เหลือง/แดงตามระดับความเสี่ยง)
    st.subheader("ผลการทำนาย")
    msg = f"ระดับความเสี่ยงในการลงทุนที่ทำนายได้: **{pred_th}** ({pred_label}) — โมเดล: {model_name}"
    if pred_label == "Low":
        st.success(msg)
    elif pred_label == "Medium":
        st.warning(msg)
    else:
        st.error(msg)

    # 7.5 แสดงความมั่นใจของโมเดลในแต่ละระดับ
    prob_df = pd.DataFrame(
        {
            "ระดับความเสี่ยง": [RISK_TEXT.get(c, c) for c in class_values],
            "ความน่าจะเป็น": [float(p) for p in probs],
        }
    ).set_index("ระดับความเสี่ยง")
    st.bar_chart(prob_df)
    st.dataframe(prob_df.style.format("{:.1%}"))

    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(df)
