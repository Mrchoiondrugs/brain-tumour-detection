import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Brain Tumor Detector", page_icon="🧠", layout="centered"
)

st.title("🧠 Brain Tumor Detection App")
st.write(
    "Upload a brain MRI scan image below to detect and localize potential"
    " tumors using a fine-tuned YOLO model."
)


@st.cache_resource
def load_yolo_model(model_path):
  return YOLO(model_path)


try:
  model = load_yolo_model("btumour.pt")
except Exception as e:
  st.error(
      "Error loading model 'btumour.pt'. Please make sure the file is in the"
      " same directory as app.py."
  )

uploaded_file = st.sidebar.file_uploader(
    "Choose an MRI Image...", type=["jpg", "jpeg", "png"]
)

# Predict button in the sidebar
predict_btn = st.sidebar.button("Predict Tumor", type="primary")

CONF_THRESHOLD = 0.25

if uploaded_file is not None:
  input_image = Image.open(uploaded_file)

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Uploaded Scan")
    st.image(input_image, use_container_width=True)

  # Only run prediction when button is clicked
  if predict_btn:
    results = model.predict(source=input_image, conf=CONF_THRESHOLD)

    res = results[0]

    # conf=False hides the confidence decimals on top of the image bounding boxes
    im_array = res.plot(conf=False)
    annotated_img = Image.fromarray(im_array[..., ::-1])

    with col2:
      st.subheader("Detection Result")
      st.image(annotated_img, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Summary")

    if len(res.boxes) == 0:
      st.warning("No tumor detected in the scan.")
    else:
      st.success(f"Detected **{len(res.boxes)}** region(s):")
      for idx, box in enumerate(res.boxes, start=1):
        cls_id = int(box.cls[0])
        label_name = res.names[cls_id]
        st.write(f"**{idx}. {label_name}**")
  else:
    with col2:
      st.subheader("Detection Result")
      st.info("Click 'Predict Tumor' in the sidebar to run analysis.")
else:
  st.info("👈 Please upload an MRI scan from the sidebar to begin.")

  

