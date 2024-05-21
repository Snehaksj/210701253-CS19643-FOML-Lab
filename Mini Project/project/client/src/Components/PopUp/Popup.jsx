import styles from "./popupdialog.module.css";
export default function Popup({ message, onClose }) {
  return (
    <div className={styles.popup}>
      <div className={styles.popupContent}>
        <img src="/Caution.svg" alt="Caution" />
        <p>{message}</p>
        <button onClick={onClose}>Ok</button>
      </div>
    </div>
  );
}
