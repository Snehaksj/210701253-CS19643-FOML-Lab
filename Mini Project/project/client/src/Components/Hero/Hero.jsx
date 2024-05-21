import Caption from "./Caption/Caption.jsx";
import Floatingimage from "./Floating image/Floatingimage";
import styles from "./Hero.module.css";
import Upload from "./Upload/Upload.jsx";
export default function () {
  return (
    <div className={styles.hero}>
      <div className={styles.subhero}>
        <Caption />
        <Floatingimage/>
      </div>
      <div className={styles.subhero}>
        <Upload/>
      </div>
    </div>
  );
}
