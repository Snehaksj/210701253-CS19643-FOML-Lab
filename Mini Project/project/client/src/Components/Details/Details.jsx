import React from "react";

export default function Details({ data }) {
  return (
    <div>
      <h2>Details Page</h2>
      {data.map((item, index) => (
        <div key={index}>
          <p>Name: {item.name}</p>
          <p>Area: {item.area_found}</p>
        </div>
      ))}
    </div>
  );
}
