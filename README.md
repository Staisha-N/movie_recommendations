# NeuralFlix: Smart Movie Suggestions

NeuralFlix is a neural network powered application that creates customized movie recommendation for you. It uses a Go web-scraper to determine relationships between age groups and movie genre preferences. Pytorch considers this data along with movie preferences of people of your same age and gender to determine the best recommendations for you.

## Installation

This application uses Go, Python and Python libraries. Python and Go should be installed and ensure you have the necessary packages.

```bash
pip install numpy pandas torch torchvision scikit-learn
pip install Flask
go get github.com/PuerkitoBio/goquery
```

## Usage

First, run the web scraper in Go from the web_scraper folder:

```go
cd web_scraper
go run main.go
```

Next, run the Flask web application from the root directory:

```python
cd ../
python app.py
```

Then open up the development server at http://127.0.0.1:5000 to use the app.
