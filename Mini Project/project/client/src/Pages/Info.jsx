import React, { useEffect, useState } from "react";
import axios from "axios";
import { useParams } from "react-router-dom";
import Background from "../Components/Background/Background";
import Navbar from "../Components/NavigationBar/Navbar";
// import Corousal from "../Components/Corousal/Card";
import Card from "../Components/Card/Card";
// import Details from "../Components/Details/Details"; // Import Details component

export default function Info() {
  const [plantDetails, setPlantDetails] = useState([]);
  const {
    name1,
    name2,
    name3,
    name4,
    name5,
    percentage1,
    percentage2,
    percentage3,
    percentage4,
    percentage5,
  } = useParams();

  // Create an object containing the parameters
  const data = {
    plant_names: [name2, name1, name3],
  };
  const match = {
    plant_match: [
      // percentage4,
      percentage2,
      percentage1,
      percentage3,
      // percentage5,
    ],
  };

  return (
    <Background>
      <Navbar />
      {/* Render plant details using the Details component */}
      <Card data={data} match={match} />
    </Background>
  );
}
