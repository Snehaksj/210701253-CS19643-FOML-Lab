import React from "react";
import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import Background from "../Components/Background/Background";
import Navbar from "../Components/NavigationBar/Navbar";
import styles from "./Page_css/Plant.module.css";

export default function Plant() {
  const [plantDetails, setPlantDetails] = useState([]);
  const { name } = useParams();
  const data = {
    plant_name: name,
  };
  const getImageSrc = (name) => {
    // Replace spaces with underscores and append ".jpg" extension
    return `/${name.replace(/\s+/g, "_")}.jpg`;
  };
  useEffect(() => {
    async function fetchDetails() {
      try {
        const response = await axios.post("http://127.0.0.1:5000/fetch", data, {
          headers: {
            "Content-Type": "application/json",
          },
        });
        setPlantDetails(response.data.plant_detail);
        console.log(response.data.plant_detail);
      } catch (error) {
        console.error("Error fetching plant details:", error);
      }
    }
    fetchDetails();
  }, []);
  // Render plant details if plantInfo is defined
  return (
    <Background className={styles.bg}>
      <Navbar />
      <div className={styles.container}>
        <div className={styles.subcontainer}>
          <h3>{plantDetails.name}</h3>
          <p>
            <em>{plantDetails.binomial_nomenclature}</em>
          </p>
          <p>
            <b>Description:</b>
          </p>
          <p>{plantDetails.description}</p>
          <p>
            <b>Area Found:</b>
          </p>
          <p>{plantDetails.area_found}</p>
          <p>
            <b>Best Suitable Soil:</b>
          </p>
          <p>{plantDetails.best_suitable_soil}</p>
          <p>
            <b>Medicinal values:</b>
          </p>
          <p>{plantDetails.medicinal_values}</p>
        </div>
        <div className={styles.imgcontainer}>
          <img src={getImageSrc(name)} alt={name} />
        </div>
      </div>
    </Background>
  );
}
