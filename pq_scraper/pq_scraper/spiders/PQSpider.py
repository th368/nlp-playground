import re
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class PQSpider(CrawlSpider):
    name = "PQSpider"

    allowed_domains = ["questions-statements.parliament.uk"]
    start_urls = ["https://questions-statements.parliament.uk/written-questions?SearchTerm=&DateFrom=01%2F01%2F1990&DateTo=29%2F10%2F2021&AnsweredFrom=&AnsweredTo=&House=Commons&Answered=Answered&Expanded=True"] # this is the full set of PQs on record...

    rules = (
        Rule(LinkExtractor(allow='written-questions', deny='detail')),
        Rule(LinkExtractor(allow='detail'), callback='parse_item'),
    )

    def parse_item(self, response):

        # pull out department info
        department = re.sub('(?i)question for ', '' , response.xpath('//*[@id="main-content"]/div[1]/div/div[2]/div/div/h2/text()').get().strip())

        # pull out our answer text
        # the question seems to be housed within "col-lg-12 read-more"
        answer_text = response.xpath('//*[@class="col-lg-12"]/p//text()')
        answer_date = answer_text[0].get().strip()
        answer = answer_text[1:].getall()

        # pull name and political party
        # questioner is always the first name returned, respondent the second.
        mp_names = [response.xpath('//*[@class="primary-info"]/text()')[num].get().strip() for num in range(2)]
        political_parties = [response.xpath('//*[@class="tertiary-info"]/text()')[num].get().strip() for num in range(2)]

        if answer == "":
            pass
        else:
            yield {
                'department': department,
                'question': response.xpath('//*[@id="collapse-details"]/p/text()').get().strip(), # this seems to be consistent across all pages
                'answer': "\n".join([a.strip() for a in answer]), # might want to change the join to ""
                'url': response.url,
                'questioner': mp_names[0],
                'question_party': political_parties[0],
                'respondent': mp_names[1],
                'respondent_party': political_parties[1],
                'answer_date': answer_date
            }
