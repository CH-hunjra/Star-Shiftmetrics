This is a personal coding project, aligned with a passion of mine: Space, and the redshift of celestial objects.

Star Shiftmetrics is, in general terms, a predictively modelled data-visualiser. More specifically, one focused on the visualisation and manipulation of over 70,000 Stars, Quasars and Galaxies.

Explore the accuracy of specific models in processing and predicting the redshift of celestial objects. Observe and learn from Colour Index graphs, discovering Quasars and Galaxies billions of years old hidden in clusters of Stars.

The application is a WIP, being created using the following languages and libraries:
- Python
    - pandas
    - scikit-learn
    - numpy
- Dash
    - Utilising HTML & CSS within a python environment.
    - plotly
- LaTeX

Prerequisites to run:
- Minimum of Python 3.13, higher recommended.
- Internet Connection for celestial lookup redirects.

Current Features:
- Overview: My reasons for creating the project, and what I aim to portray and achieve through it.

- Predictive Modelling: This is the main section, it contains all the current models, each trained on SDSS data, each visualising the data in a different way, and predicting redshift at different accuracy levels. Under each model is the following:
    - An overview of the model.
    - Model Performance data (MSE and R²)
    - True VS Predicted Redshift: A detailed, interactive graph comparing true redshift versus the models predicted redshift. visually portraying accuracy.
    - Colour Index Exploration: A deep-diving, interactive graph which compares different wavelengths of light and what patterns and understandings they show across 70,000+ data points.
    - Lookup Table: A table that shows the selected celestial objects information, alongside personal backend calculations, an accurate Lookback Age calculation, alongside a link to the objects SDSS data and map visuals.

- Calculations: A deep-dive into the calculations used for this project, alongside examples written in LaTeX.

- FAQ: Answers basic questions one may ask after exploring the application.