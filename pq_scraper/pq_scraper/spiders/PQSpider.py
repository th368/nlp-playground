import re
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class PQSpider(CrawlSpider):
    name = "PQSpider"

    allowed_domains = ["questions-statements.parliament.uk"]
    start_urls = ["https://questions-statements.parliament.uk/written-questions?SearchTerm=&DateFrom=01%2F01%2F1990&DateTo=29%2F10%2F2021&AnsweredFrom=&AnsweredTo=&House=Bicameral&Answered=Answered&Expanded=True"] # this is the full set of PQs on record...


    rules = (
        Rule(LinkExtractor(allow='written-questions', deny='detail')),
        Rule(LinkExtractor(allow='detail'), callback='parse_item'),
    )

    def parse_item(self, response):
        
        # extract uin
        uin = response.xpath('//*[@id="main-content"]/div[1]/div/div[2]/div/div/p/text()').get().strip()
        uin = re.search("(?<=UIN )[0-9|a-z|A-Z]+", uin).group(0) # strip additional text

        # pull out department info
        department = re.sub('(?i)question for ', '' , response.xpath('//*[@id="main-content"]/div[1]/div/div[2]/div/div/h2/text()').get().strip())

        # pull out our answer text
        # the question seems to be housed within "col-lg-12 read-more"
        answer_text = response.xpath('//*[@class="col-lg-12"]/p//text()')
        answer = answer_text[1:].getall()

        # if the question has been withdrawn or is yet to be answered, then just skip over it
        if len(answer)==0:
            print("Withdrawn or other...")
            pass
        else:
            # extract answer date
            answer_date = answer_text[0].get().strip()

            # pull name and political party
            # questioner is always the first name returned, respondent the second.
            mp_names = [response.xpath('//*[@class="primary-info"]/text()')[num].get().strip() for num in range(2)]
            political_parties = [response.xpath('//*[@class="tertiary-info"]/text()')[num].get().strip() for num in range(2)]
            constituency = [response.xpath('//*[@class="indicator indicator-label"]/text()')[num].get().strip() for num in range(2)]
            house = response.xpath('//*[@class="info-inner"]/div[2]/div[1]/text()').getall()
            
            yield {
                'department': department,
                'uin': uin,
                'hansard_header': response.xpath('//*[@id="main-content"]/div[1]/div/div[1]/div[1]/div/h1/text()').get().strip(), # add .strip() as a pipeline at some point...
                'question': response.xpath('//*[@id="collapse-details"]/p/text()').get().strip(), # this seems to be consistent across all pages
                'answer': "\n".join([a.strip() for a in answer]), # might want to change the join to ""
                'url': response.url,
                'questioner': mp_names[0],
                'question_party': political_parties[0],
                'questioner_constituency': constituency[0],
                'respondent': mp_names[1],
                'respondent_party': political_parties[1],
                'respondent_constituency': constituency[1],
                'answer_date': answer_date,
                'house': "".join([h.strip() for h in set(house)]) 
            }
