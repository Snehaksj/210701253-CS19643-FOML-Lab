import styles from "./Page_css/About.module.css";
import Navbar from "../Components/NavigationBar/Navbar.jsx";
import Banner from "../Components/Banner/Banner.jsx";
export default function About() {
  return (
    <div className={styles.about}>
      <Navbar />
      <Banner></Banner>
      <div className={styles.description}>
        <p>
          Welcome to AyurBotanica, a project born from the passion of college
          students exploring the wonders of natural remedies and wellness. We
          believe in the power of plants to enhance health and happiness.
        </p>
        <h3>Our Mission</h3>
        <p>
          Our mission is to blend traditional botanical knowledge with modern
          understanding to create simple, effective, and affordable natural
          solutions for everyday well-being.
        </p>
        <h3>What Drives Us?</h3>
        <ul>
          <li>
            <p>
              Curiosity: We are driven by curiosity and a desire to learn,
              experiment, and discover the potential of botanicals.
            </p>
          </li>
          <li>
            <p>
              Innovation: We strive to innovate and explore creative ways to
              integrate natural remedies into daily life.
            </p>
          </li>
          <li>
            <p>
              Community: We aim to build a supportive community of students and
              enthusiasts interested in holistic living and wellness.
            </p>
          </li>
        </ul>
        <h3>Why Choose AyurBotanica?</h3>
        <ul>
          <li>
            <p>
              Affordability: Our focus is on creating affordable solutions that
              are accessible to everyone, especially students on a budget.
            </p>
          </li>
          <li>
            <p>
              Simplicity: We believe in keeping things simple, offering
              easy-to-understand products and resources for natural living.
            </p>
          </li>
          <li>
            <p>
              Learning Experience: AyurBotanica is not just about products; it's
              also a learning journey where students can explore and grow their
              understanding of herbal remedies.
            </p>
          </li>
        </ul>
        <p>
          Join us on our journey as we explore the fascinating world of
          AyurBotanica together!
        </p>
      </div>
    </div>
  );
}
