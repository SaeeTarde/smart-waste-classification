const predictButton = document.getElementById("predictBtn");
const imageInput = document.getElementById("imageInput");
const resultDiv = document.getElementById("result");

predictButton.addEventListener("click", async () => {
  if (imageInput.files.length === 0) {
    alert("Please select an image first");
    return;
  }

  const file = imageInput.files[0];

  const formData = new FormData();
  formData.append("file", file);

  resultDiv.innerHTML = "⏳ Predicting...";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Server error");
    }
    const data = await response.json();

    // warning section (only if confidence is low)
    let warningHTML = "";
    if (data.warning) {
      warningHTML = `<p style="color:red;"><strong>⚠️ Warning:</strong> ${data.warning}</p>`;
    }

    resultDiv.innerHTML = `
      <h3>Prediction Result</h3>

      <p><strong>Waste Type:</strong> ${data.waste_type}</p>
      <p><strong>Confidence:</strong> ${data.confidence}%</p>

      ${warningHTML}

      <p><strong>Disposal Method:</strong> ${
        data.recommendation?.disposal_method || "Not available"
      }</p>
      <p><strong>Eco Score:</strong> ${data.recommendation.eco_score}</p>
      <p><strong>Tip:</strong> ${data.recommendation.tip}</p>
    `;
  } catch (error) {
    console.error(error);
    resultDiv.innerHTML = "❌ Error connecting to backend.";
  }
});
