import Background from "../Components/Background/Background.jsx";
import Navbar from "../Components/NavigationBar/Navbar.jsx";
import Hero from "../Components/Hero/Hero.jsx";
function Home() {
  return (
    <div className="Home">
      <Background>
        <Navbar></Navbar>
        <Hero></Hero>
      </Background>
    </div>
  );
}

export default Home;
