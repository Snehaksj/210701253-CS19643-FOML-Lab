import React, { useState, useEffect } from "react";
import styles from "./Card.module.css";
import { Link } from "react-router-dom";

export default function Card({ data, match }) {
  const [firstLoad, setFirstLoad] = useState(true);
  const { plant_names } = data;
  const { plant_match } = match;

  useEffect(() => {
    setFirstLoad(true);
  }, []);

  const getImageSrc = (name) => {
    return `/${name.replace(/\s+/g, "_")}.jpg`;
  };

  const handleMiddle = (isMiddleCard) => {
    if (isMiddleCard && firstLoad) {
      setFirstLoad(false);
    }
  };

  const centerIndex = Math.floor(plant_names.length / 2);

  return (
    <>
      <p className={styles.heading}>Plant Info</p>
      <div className={styles.card}>
        {plant_names.map((name, index) => {
          const isCenterCard = index === centerIndex;
          const isMiddleCard = Math.abs(index - centerIndex) === 1;
          const cardClass = isCenterCard
            ? styles.centerCard
            : isMiddleCard
            ? styles.middleCard
            : styles.carditems;

          return (
            <Link
              to={`/Plant/${name}`}
              key={index}
              className={cardClass}
              onMouseEnter={() => handleMiddle(isMiddleCard)}
            >
              <img
                src={getImageSrc(name)}
                alt={name}
                className={
                  firstLoad && isCenterCard
                    ? `${styles.firststepimg} ${styles.centerCardimg}`
                    : isCenterCard
                    ? styles.centerCardimg
                    : isMiddleCard
                    ? styles.middleCardimg
                    : ""
                }
              />
              <p
                className={
                  firstLoad && isCenterCard ? styles.firststep : styles.caption
                }
              >
                {name}
                <br />
                {plant_match[index]}%
              </p>
            </Link>
          );
        })}
      </div>
    </>
  );
}
