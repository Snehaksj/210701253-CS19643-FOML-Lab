import React, { useState } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import Popup from "../../PopUp/Popup";
import styles from "./Upload.module.css"; // Import the CSS file
import Caption from "../Caption/Caption";
import style from "../Caption/Caption.module.css";
import Prediction from "../../Prediction/Prediction";
const Upload = () => {
  const [SelectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [showPopup, setShowPopup] = useState(false);
  const [apiResponse, setApiResponse] = useState(null);
  const [apiResponseBool, setapiResponseBool] = useState(false);
  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0]; // Accept only the first file if multiple files are dropped
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file)); // Generate preview URL
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1, // Allow only one file to be dropped
    accept: "image/*", // Accept only image files
  });
  const handleUpload = async () => {
    if (!SelectedFile) {
      setShowPopup(true); // Show the pop-up if no file is selected
      return;
    }

    const formData = new FormData();
    formData.append("image", SelectedFile);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("Upload successful:", response.data);
      setapiResponseBool(true);
      setApiResponse(response.data); // Store API response in state variable
      // Handle successful upload
    } catch (error) {
      console.error("Upload error:", error);
      // Handle upload error
    }
  };
  return (
    <div className={styles.upload}>
      {/* <div className={style.caption}>
        <p>
          Unveiling the Healing Secrets of Medicinal Plants <br />
          with Deep Learning <br />
          What are you waiting for? Try out now!
        </p>
      </div> */}
      <div {...getRootProps()} className={styles.dropZone}>
        <input {...getInputProps()} />
        {previewUrl && (
          <img src={previewUrl} alt="Preview" className={styles.preview} />
        )}
        {/* Display preview image */}
        {!previewUrl && (
          <>
            <img src="/cloud.svg" alt="Cloud" className={styles.cloud} />
            <p>
              <b>Drop</b> or <b>Upload</b> an Image
            </p>
          </>
        )}
      </div>
      <button className={styles.button} onClick={handleUpload}>
        PREDICT
      </button>
      {showPopup && (
        <>
          <Popup
            message="Please select a file."
            onClose={() => setShowPopup(false)} // Close the pop-up when the user clicks "Close"
          />
        </>
      )}
      {apiResponseBool && (
        <Prediction
          Predict={apiResponse}
          onClose={() => setapiResponseBool(false)}
        />
      )}
    </div>
  );
};
export default Upload;
