import Navbar from "../Components/NavigationBar/Navbar.jsx";
import styles from "./Page_css/Contact.module.css";
export default function Contact() {
  return (
    <div className={styles.contact}>
      <Navbar></Navbar>
      <div className={styles.contactbanner}>
        <h2 className={styles.heading}>Contact us</h2>
      </div>
      <p className={styles.passage}>
        Don't hesitate to reach out to us; we're a small team passionate about
        what we do! Whether you have questions, suggestions, or just want to
        chat, we're here for you. As a project run by just two individuals, we
        value each interaction and strive to provide the best support possible.
        Feel free to drop us a message using the contact details provided below.
        We appreciate your interest and look forward to hearing from you!
      </p>
      <h3 className={styles.imgheading}>Meet the Team</h3>
      <div className={styles.images}>
        <div className={styles.img1details}>
          <div className={styles.img1}></div>
          <p className={styles.name}>A.SATYANARAYANA</p>
          <p className={styles.details}>Prefinal Year Student</p>
          <p className={styles.details}>CSE Department</p>
          <p className={styles.details}>REC</p>
          <div className={styles.icons}>
            <a href="mailto:210701235@rajalakshmi.edu.in">
              <img src="/gmail.svg" alt="gmail" className={styles.icon} />
            </a>
            <a href="https://www.linkedin.com/in/satyanarayanaa/">
              <img src="/linkedin.svg" alt="linkedin" className={styles.icon} />
            </a>
            <a href="https://github.com/CrypticFate5">
              <img src="/github.svg" alt="github" className={styles.icon} />
            </a>
          </div>
        </div>
        <div className={styles.img2details}>
          <div className={styles.img2}></div>
          <p className={styles.name}>K.S.J.SNEHA</p>
          <p className={styles.details}>Prefinal Year Student</p>
          <p className={styles.details}>CSE Department</p>
          <p className={styles.details}>REC</p>
          <div className={styles.icons}>
            <a href="mailto:210701253@rajalakshmi.edu.in">
              <img src="/gmail.svg" alt="gmail" className={styles.icon} />
            </a>
            <a href="https://www.linkedin.com/in/snehaksj/">
              <img src="/linkedin.svg" alt="linkedin" className={styles.icon} />
            </a>
            <a href="https://github.com/Snehaksj">
              <img src="/github.svg" alt="github" className={styles.icon} />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
