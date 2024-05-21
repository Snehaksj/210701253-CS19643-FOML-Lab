import Home from "./Pages/Home.jsx";
import About from "./Pages/About.jsx";
import Contact from "./Pages/Contact.jsx";
import Info from "./Pages/Info.jsx";
import Plant from "./Pages/Plant.jsx";
import { BrowserRouter, Routes, Route } from "react-router-dom";
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/contactus" element={<Contact />} />
        <Route
          path="/info/:name1/:percentage1/:name2/:percentage2/:name3/:percentage3/:name4/:percentage4/:name5/:percentage5"
          element={<Info />}
        />
        <Route path="/Plant/:name" element={<Plant />}></Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
