import scrapy
import numpy as np

GOV_UK_URL = "www.gov.uk"
MOJ_URL = "https://www.gov.uk/search/research-and-statistics?content_store_document_type=statistics_published&organisations%5B%5D=ministry-of-justice&order=updated-newest"


class moj_pub_scraper(scrapy.Spider):
    name = "moj_pub_spider"
    start_urls = [MOJ_URL]


    def parse(self, response):

        # create our url hub
        url_hub = response.xpath('//*[@id="js-results"]/div/ul/li//@href').getall()

        for url in url_hub:
            yield response.follow(url, callback = self.parse_pub_page)

        # find next page tag -- need to get this working...
        next_page = response.xpath('//*[@id="js-pagination"]/nav/ul/li/a/@href')[-1].get()
        if next_page is not None:
            yield response.follow(next_page, callback = self.parse)


    def parse_pub_page(self, response):
        # pull out the classes for each object on our publication page
        article_classes = np.array(response.xpath('//*[@id="documents"]/div//p/span[1]//text()').getall())
        links = np.array(response.xpath('//*[@id="documents"]/div//h3/a/@href').getall())
        html_links = links[np.where(article_classes == 'HTML')]

        if len(html_links) > 0:
            for url in html_links:
                yield response.follow(url, callback = self.parse_pub)
        
    def parse_pub(self, response):
        
        yield {
            'text': [item.strip() for item in response.xpath('//*[@id="contents"]/div[3]/div/div/div//text()').getall()]
        }