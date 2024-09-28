package main

import (
    "encoding/csv"
    "fmt"
    "log"
    "net/http"
    "os"
    "strings"

    "github.com/PuerkitoBio/goquery"
)

func main() {
    //Scrape this website about movie genres and demographics
    url := "https://filmgrail.com/blog/cinema-audience-demographics-analysis-insights-and-data/"

    //Fetch webpage
    res, err := http.Get(url)
    if err != nil {
        log.Fatal(err)
    }
    defer res.Body.Close()

    if res.StatusCode != 200 {
        log.Fatalf("Failed to fetch URL: %s, Status Code: %d", url, res.StatusCode)
    }

    doc, err := goquery.NewDocumentFromReader(res.Body)
    if err != nil {
        log.Fatal(err)
    }

    //Create a CSV with the age, movie genre relationship in the data folder
    file, err := os.Create("../data/age_genre_relations.csv")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    writer := csv.NewWriter(file)
    defer writer.Flush()

    writer.Write([]string{"Age", "Genre"})

    //Scrape data
    doc.Find("tbody tr").Slice(0, 3).Each(func(index int, element *goquery.Selection) {
        ageGroup := element.Find("td").Eq(0).Text() //Get the ages
        genre := element.Find("td").Eq(1).Text() //Get the movie genre preferences for the group

        ages := strings.Split(ageGroup, "-")
        var startAge, endAge int
        if (ageGroup == "40+") {
            startAge = 40
            endAge = 60
        } else {
            if len(ages) == 2 {
                fmt.Sscanf(ages[0], "%d", &startAge)
                fmt.Sscanf(ages[1], "%d", &endAge)
            }
        }
        
        //Write each age with its genre to CSV
        for age := startAge; age <= endAge; age++ {
            writer.Write([]string{fmt.Sprint(age), genre})
        }
    })

    fmt.Println("Data has been written to age_genre_relations.csv")
}
