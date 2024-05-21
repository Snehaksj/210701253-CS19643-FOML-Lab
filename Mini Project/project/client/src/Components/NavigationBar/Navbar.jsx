import styles from "./Navbar.module.css";
import { Link } from "react-router-dom";
export default function Navbar() {
  return (
    <nav className={styles.navbar}>
      <div>
        <h1>AyurBotanica</h1>
        <img src="/favicon.png" alt="logo" />
      </div>
      <ul>
        <li>
          <Link to="/">Home</Link>
        </li>
        <li>
          <Link to="/about">About us</Link>
        </li>
        <li>
          <Link to="/contactus">Contact us</Link>
        </li>
      </ul>
    </nav>
  );
}
