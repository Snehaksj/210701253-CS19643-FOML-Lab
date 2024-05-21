import styles from "./Caption.module.css";
import Upload from "../Upload/Upload";
export default function Caption() {
  return (
    <div className={styles.caption}>
      <p className={styles.heading}>
        AYURBOTANICA: <br />
        BOTANIC DISCOVERY
      </p>
      <p
        style={{ marginTop: -2 + "vh", fontSize: 1.5 + "rem", fontWeight: 450 }}
      >
        Unveiling the Healing Secrets of Medicinal Plants with Deep Learning.{" "}
        <br />
        What are you waiting for? Try out now!
      </p>
    </div>
  );
}
