import { Link } from "react-router-dom";
import styles from "./Prediction.module.css";
import React, { useState, useEffect } from "react";

export default function Prediction({ Predict, onClose }) {
  const dataArray = [];

  // Traverse and accumulate the values into dataArray
  Object.values(Predict).forEach((innerArray) => {
    innerArray.forEach((item) => {
      const name =
        item[1].split("_")[0].charAt(0).toUpperCase() +
        item[1].split("_")[0].substr(1);
      const percentage = (item[0] * 100).toFixed(2);
      dataArray.push([parseFloat(percentage), name]);
    });
  });
  const jsonData = { pred: dataArray };

  const formattedData = {};
  jsonData.pred.forEach((item, index) => {
    const percentageKey = `percentage${index + 1}`;
    const nameKey = `name${index + 1}`;
    formattedData[percentageKey] = item[0];
    formattedData[nameKey] = item[1];
  });

  console.log(formattedData);
  return (
    <div className={styles.Prediction}>
      <div className={styles.PredictionContent}>
        <img src="/favicon.png" alt="" />
        <div>
          {Object.entries(Predict).map(([key, innerArray]) => (
            <div key={key}>
              {innerArray.map((item, innerIndex) => (
                <div key={innerIndex}>
                  <li>
                    {item[1].split("_")[0].charAt(0).toUpperCase() +
                      item[1].split("_")[0].substr(1)}{" "}
                    Leaf: {(item[0] * 100).toFixed(2)} %
                  </li>
                  <br />
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className={styles.buttons}>
          <button onClick={onClose}>Ok</button>
          <Link
            to={`/info/${formattedData.name1}/${formattedData.percentage1}/${formattedData.name2}/${formattedData.percentage2}/${formattedData.name3}/${formattedData.percentage3}/${formattedData.name4}/${formattedData.percentage4}/${formattedData.name5}/${formattedData.percentage5}`}
          >
            <button>Know more</button>
          </Link>
        </div>
      </div>
    </div>
  );
}
