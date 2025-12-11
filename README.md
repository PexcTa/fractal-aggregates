<a id="readme-top"></a>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


## About The Project

This repo is a Streamlit-hosted app that allows the user to build simple, geometrical, three-dimensional models of mass fractal aggregates. It was developed to assist with small angle scattering data analysis. The parameter range is limited in the app in order to keep simulations reasonably fast; therefore, the app is a bit of a demo for the actual implementation. 

The app itself can be accessed here: [kramar-pemfa.streamlit.app](https://kramar-pemfa.streamlit.app/)

This is a work in progress.


## Getting Started

The app is built using a few standard Python packages. You can access the current version of the app on Streamlit. If you want to run and host it yourself, here are some basic instructions. These assume you are using the Anaconda disribution of Python. 

* Clone the repository
    ```sh
    git clone https://github.com/yourusername/fractal-aggregates.git
    cd fractal-aggregates
    ```

* Create and activate a conda environment
    ```sh
    git clone https://github.com/PexcTa/fractal-aggregates.git
    cd fractal-aggregates
    ```

* Install the dependencies (streamlit, numpy, matplotlib, plotly, scipy)
    ```sh
    pip install -r requirements.txt
    ```


## Usage

An example will be added here.


## Roadmap

- Add the ability to generate agglomerates out of aggregates
- Add the ability to generate polydisperse aggregates
- For future releases, add other algorithms (e.g. DLA)


## Contributing

All contributions and comments are greatly appreciated.


## License

Distributed under the The Unlicense. 


## Contact

Boris V. Kramar - borisvokramar@gmail.com

Project Link: [https://github.com/PexcTa/fractal-aggregates](https://github.com/PexcTa/fractal-aggregates)


## Acknowledgments

* I am using an adapted version of othnealdrew's [Best-README-Template](https://github.com/othneildrew/Best-README-Template/blob/main/BLANK_README.md)
* The algorithm used by the app was described by Guesnet et al. in a scientific publication at [10.1016/j.physa.2018.07.061](https://doi.org/10.1016/j.physa.2018.07.061)
* All code was developed with assistance from [Qwen3-Max](https://chat.qwen.ai)

<p align="right">(<a href="#readme-top">back to top</a>)</p>