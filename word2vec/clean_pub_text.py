import json
import re
import string

pub_data_filepath = "../moj_pub_scraper/pub_text.json"
# Opening JSON file
f = open(pub_data_filepath, )

# returns JSON object as a dictionary
pub_data = json.load(f)

# pull out our text and join it together
full_text = pub_data[0]['text']
full_text = ' '.join(full_text)


# remove links, incl. those in paranthesis
re.sub(r"\S*https?:\S*", "", full_text)
# remove 'Main Points' header
re.sub(r"(?i)Main Points", "", full_text)
# remove unicodes (and any other unprintable characters)
printable = set(string.printable)
''.join(filter(lambda x: x in printable, full_text))
# another way of doing the above...
# f = re.sub(r'[^%s%s]' % (re.escape(string.punctuation), 'A-Za-z0-9 '), '', full_text)
# strip whitespace
full_text.strip()

def clean_pub_text(text):
    # remove links, incl. those in paranthesis
    text = re.sub(r"\S*https?:\S*", "", text)
    # remove 'Main Points' header
    text = re.sub(r"(?i)Main Points", "", text)
    # remove unicodes (and any other unprintable characters)
    text = ''.join(filter(lambda x: x in set(string.printable), text))
    # another way of doing the above...
    # f = re.sub(r'[^%s%s]' % (re.escape(string.punctuation), 'A-Za-z0-9 '), '', full_text)
    # strip whitespace
    text = text.strip()

    # let's finalise this by adding in some custom tokens for numbers and percents
    text = re.sub(r'[0-9]+-', '[:NUM:] ', text) # replace [num]-month with token
    text = re.sub(r'[\d|\\.]*%', '[:PERC:]', text) # replace %s with token
    text = re.sub(r'(?<=\s)[\d|\.|,]+\S{1}', '[:NUM:]', text) # replace numbers with token
    text = re.sub(r'\S*@\S*\s?', '', text) # remove emails
    # and finally, to allow me to reuse old code,
    # add a line break whenever we have a valid fullstop
    text = re.sub('\.{1}\s{1}', '.\n ', text)

    return(text)

# write to moj_pub.txt
with open('moj_pub.txt', 'w') as f:
    f.write(clean_pub_text(full_text))
with open('../full_text.txt', 'w') as f:
    f.write(full_text)



# test_string = "4.5555. Testing to see if it correctly. pulls out fullstops."
# re.sub('\.{1}\s{1}', '.\n ', test_string)
#
# test_string = "previous 12-month period"
# re.sub('[0-9]+-', '[:NUM:] ', test_string)
#
#
# test_string = "100%  previous 12-month period, 5.444%, 5555%, 5555, COVID-19"
# # re.sub('[\d|\.]+\S{1}', '[:NUM:]', test_string)
# re.sub('(?<=\s)[\d|\.]+\S{1}', '[:NUM:]', test_string)
#
# re.findall('(\d)*%', test_string)
# re.findall('[\d]*%', test_string)
#
# # re.findall("(?P<url>https?://[^\s]+)", full_text)
#
# # def clean_pub_text(full_text):
#
# # ascii strip should be:
# # \\\u\S+
# re.findall("((www\.|http://|https://)(www\.)*.*?(?=(www\.|http://|https://|$)))", full_text)

# def custom_standardization(input_data):
#   lowercase = tf.strings.lower(input_data)
#   return tf.strings.regex_replace(lowercase,
#                                   '[%s]' % re.escape(string.punctuation), '')
