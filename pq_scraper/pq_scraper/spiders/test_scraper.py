import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

# question by:
# response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[1]/div/a/div/div[1]/div[2]/text()').get().strip()

# question party:
# response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[1]/div/a/div/div[1]/div[3]/text()').get().strip()

# respondent:
# response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[3]/div/a/div/div[1]/div[2]/text()').get().strip()

# respondent party:
# response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[3]/div/a/div/div[1]/div[3]/text()').get().strip()

# question:
# response.xpath('//*[@id="collapse-details"]/p/text()').get().strip()

# response:
# response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[5]/div//text()').getall()

# https://questions-statements.parliament.uk/written-questions/detail/2021-10-21/60506



class test_scraper(scrapy.Spider):
    name = "test_scraper"
    start_urls = ["https://questions-statements.parliament.uk/written-questions"]
    # start_urls = ["https://questions-statements.parliament.uk/written-questions/detail/2021-05-14/1357"]

    # rules = (
    #     Rule(LinkExtractor(allow='written-questions', deny='detail')),
    #     Rule(LinkExtractor(allow='written-questions'), callback='parse_item'),
    # )

    def parse(self, response):

        urls = response.xpath('//*[@id="main-content"]/div/article/div/div[2]/div[2]//@href[contains(., "detail")]').getall()

        for url in urls:
            yield response.follow(url, callback = self.parse_scrape_pq)
    
    def parse_scrape_pq(self, response):
        # add in url, dept and fix our other info
        yield {
            'question': response.xpath('//*[@id="collapse-details"]/p/text()').get().strip(),
            'answer': response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[5]/div//text()').getall(),
            'answered': 'yes',
            'answered_dummy': 1#,
            # 'questioner': response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[1]/div/a/div/div[1]/div[2]/text()').get().strip(),
            # 'question_party': response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[1]/div/a/div/div[1]/div[3]/text()').get().strip(),
            # 'respondent': response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[3]/div/a/div/div[1]/div[2]/text()').get().strip(),
            # 'respondent_party': response.xpath('//*[@id="main-content"]/div[2]/article/div/div/div/div/div[1]/div[3]/div/a/div/div[1]/div[3]/text()').get().strip()
        }
