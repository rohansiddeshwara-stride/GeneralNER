from flair.data import Sentence
from flair.models import SequenceTagger
from flair_loder import loader


import concurrent.futures
import re
import string
import time
import stanza
import spacy



tagger = SequenceTagger.load("flair/ner-english-ontonotes")
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,ner')
nlp_spacy = spacy.load("en_core_web_lg")

from flair.data import Sentence
from flair.models import SequenceTagger





def flair_extractor(text):
  # tagger = SequenceTagger.load("flair/ner-english-ontonotes")
  sentence_flair = Sentence(text)
  tagger.predict(sentence_flair)
  entities_flair = sentence_flair.get_spans('ner')
  entity_list_flair = []
  for entity in entities_flair:
      # print("flair", [e.idx for e in entity])
      t=re.finditer(entity.text,text)
      for i in t:
        start=i.start()
        end=i.end()
        entity_list_flair.append([entity.tag, entity.text,start,end])
  return entity_list_flair


def stanza_extractor(text):
  # nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,ner')
  doc_stanza = nlp_stanza(text)
  entities_stanza = []
  for sent in doc_stanza.sentences:
      for entity in sent.ents:
          # print("stanza", entity)
          entities_stanza.append([entity.type,entity.text,entity.start_char,entity.end_char])
  return entities_stanza




def extract_email_website(text):
  email_regex= r"([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)"
  email_list=re.finditer(email_regex, text)
  emails = [["EMAIL", x.group(), x.start(), x.end()] for x in email_list]
  web_regex ="(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?([a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?"
  web_list=re.finditer(web_regex, text)
  web = [["WEB_LINK", x.group(), x.start(), x.end()] for x in web_list]
  return emails + web

def pincodes(text):
  l = []
  pincode = {"IN" : {"pin" : "\\b([1-9]{1}[0-9]{2}\\s{0,1}\\-{0,1}[0-9]{3})\\b", "len" : 6, "ll" : [744101,507130,790001,781001,800001,140119,490001,396193,362520,110001,403001,360001,121001,171001,180001,813208,560001,670001,682551,450001,400001,795001,783123,796001,797001,751001,533464,140001,301001,737101,600001,500001,799001,201001,244712,700001], "up" : [744304,535594,792131,788931,855117,160102,497778,396240,396220,110097,403806,396590,136156,177601,194404,835325,591346,695615,682559,488448,445402,795159,794115,796901,798627,770076,673310,160104,345034,737139,643253,509412,799290,285223,263680,743711], "rules" : ["10", "29", "35", "54", "55", "65", "66", "86", "87", "88", "89"]},
            "USA": {"pin" : "\\b[0-9]{5}(?:-[0-9]{4})?\\b", "len" : 5, "ll" : [501,1001,2801,3031,3901,5001,6001,7001,15001,19701,20101,20588,24701,27006,29001,30002,32003,35004,37010,38601,40003,43001,46001,48001,50001,53001,55001,57001,58001,59001,60001,63001,66002,68001,70001,71601,73001,73301,80001,82001,83201,84001,85001,87001,88901,90001,96701,97001,98001,99501], "up" : [14925,5544,2940,3897,4992,5907,6928,8989,19640,19980,24658,21930,26886,28909,29945,39901,34997,36925,38589,39776,42788,45999,47997,49971,52809,54990,56763,57799,58856,59937,62999,65899,67954,69367,71497,72959,74966,88595,81658,83414,83877,84791,86556,88439,89883,96162,96898,97920,99403,99950]},
            "GE" : {"pin" : "\\b[0-9]{5}\\b", "len" : 6, "ll" : [1067,3042,6108,7318,10115,17033,20038,21217,22844,27568,32049,34117,54290,66041,68131,80331,8056,14461,38820,98527,22041,26121,28195,40196,60306,66849,88212,49074,58084,64283], "up" : [4889,3253,6928,7989,14532,19417,21149,21789,25999,27580,33829,37299,57648,66839,79879,97909,9669,17326,39649,99998,22769,38729,28779,54585,63699,67829,89198,49849,59969,65936]},
            "SN" : {"pin" : "\\b([0-8]{1}[0-9]{5})\\b" , "len" : 6,  "ll" : [10000],"up" : [829999]},
            "UK" : {"pin" : "\\b(([A-Z]{1,2}\\d[A-Z\\d]?|ASCN|STHL|TDCU|BBND|[BFS]IQQ|PCRN|TKCA) ?\\d[A-Z]{2}|BFPO ?\\d{1,4}|(KY\\d|MSR|VG|AI)[ -]?\\d{4}|[A-Z]{2} ?\\d{2}|GE ?CX|GIR ?0A{2}|SAN ?TA1)\\b"}
           }
  for country, sub in pincode.items():
    t = re.finditer(sub["pin"], text)

    if "ll" not in sub:
      for x in t:
        l.append(["PINCODE-" + country, x.group(), x.start(), x.end()])
        # l.append([ x.start(), x.end(), x.group(), "pin-" + country])

    else:
      for x in t:
        z = int(x.group().replace("-","").replace(" ","")[0:sub["len"]])
        for i in range(len(sub["ll"])):
          if z>= sub["ll"][i] and z<= sub["up"][i]:
            l.append(["PINCODE-" + country, x.group(), x.start(), x.end()])
            # l.append([ x.start(), x.end(), x.group(), "pin-" + country])

  return l

def abbrevations(text):
  i = 0
  j = 0
  l = []
  for t in text.split(" "):
    score = 0
    score += 2 if len(t.replace(".", "")) < 6 else 0
    score += 3 if t.isupper() else 0
    score += 2 if re.search(r"(([A-Z]\.|\-){1,4}[A-Z]|\W)", t) else 0 #U.S.A
    score += 3 if re.search(r"(\([A-Z]{2,5}\))", t) else 0 # (WHO)
    score, t  = (score + 5, re.search(r"\bDr.|dr.|Mrs.|Mr.|mrs.|mr.\b", t).group())if re.search(r"\bDr.|dr.|Mrs.|Mr.|mrs.|mr.\b", t) else (score + 0, t) #Dr.
    if score > 4:
      j = i + len(t) -1
      l.append(["Abbrevation", t, i, j])
      # print(t, i, j)

    i+= len(t) + 1
  return l





def spacy_sxtractor(text):
    # nlp_spacy = spacy.load("en_core_web_lg")
    doc_spacy = nlp_spacy(text)
    entities_spacy = []
    for entity in doc_spacy.ents:
        # print("flair", [e.idx for e in entity])
        t=re.finditer(entity.text,text)
        for i in t:
          start=i.start()
          end=i.end()
          entities_spacy.append([entity.label_,entity.text,start,end])

    return entities_spacy
def list_of_lists_to_dict(list_of_lists):

    result_dict = {}

    for inner_list in list_of_lists:
      key = tuple(inner_list[2:])
      value = {"NE" : [],"text":[inner_list[1]]}

      # Check if the key already exists in the dictionary
      if key in result_dict:
          ner_list=result_dict[key]["NE"]

          if inner_list[0] not in ner_list:
            result_dict[key]["NE"].append(inner_list[0])

      else:
          result_dict[key] = value
          result_dict[key]["NE"].append(inner_list[0])



    return result_dict


def merge_overlapping_entities(input_data):
    # Create a list to store updated entries
    updated_data = []

    # Sort the keys based on their start positions
    sorted_keys = sorted(input_data.keys(), key=lambda x: x[0])

    # Iterate over the keys and values in the data
    i = 0
    while i < len(sorted_keys):
        # Initialize variables to track merged data
        merged_range = sorted_keys[i]
        merged_text = input_data[sorted_keys[i]]['text']
        merged_named_entity = input_data[sorted_keys[i]]['NE']

        # Iterate through subsequent keys to check for overlapping/intersecting ranges
        j = i + 1
        while j < len(sorted_keys):
            key2 = sorted_keys[j]
            value2 = input_data[key2]

            # Check if the ranges overlap or intersect
            if merged_range[1] >= key2[0]:
                # Update the merged_range to include the overlapping/intersecting key
                merged_range = (min(merged_range[0], key2[0]), max(merged_range[1], key2[1]))
                # Add the text from the overlapping/intersecting key to merged_text
                merged_text.extend(value2['text'])
                merged_named_entity.extend(value2["NE"])
                j += 1
            else:
                break

        # Create a new entry with the merged data
        new_entry = {merged_range: {'NE': merged_named_entity, 'text': merged_text}}

        # Append the new entry to updated_data if it's not already there
        if new_entry not in updated_data:
            updated_data.append(new_entry)

        # Move the index to the next key
        i = j

    # Create a new dictionary from the updated_data list
    result_data = {}
    for entry in updated_data:
        result_data.update(entry)

    return result_data




def filter_reduce_entity_list(d):
  for key in d.keys():
    value=d[key]["NE"]
    if len(value)>1:
        if "ORG" in value and "GPE" in value:
          d[key]["NE"]=["ORG"]
        elif "GPE" in value and "LOC" in value:
          d[key]["NE"]=["LOC"]
        elif "CARDINAL" in value and "AMOUNT" in value:
          d[key]["NE"]=['MONEY']
        elif "MONEY" in value and "AMOUNT" in value and "":
          del d[key]
        elif "ORG" in value and "EVENT" in value:
          d[key]["NE"]=['EVENT']
        elif "ABBREVATION" in value and "GPE" in value:
          d[key]["NE"]=['GPE']
        elif "ABBREVATION" in value and "ORG" in value:
          d[key]["NE"]=['ORG']
        elif "ORG" in value and 'WEB_LINK' in value:
          d[key]["NE"]=['WEB_LINK']
        elif "PERSON" in value and "ANN" in value:
          d[key]["NE"]=['PERSON']
  return d

def return_list(d):
  new_dict={}
  for k,v in d.items():
    new_dict[v["NE"][0]]={"chars":[],"text":[]}

  for k,v in d.items():
    new_dict[v["NE"][0]]["chars"].append(k)
    new_dict[v["NE"][0]]["text"].extend(v["text"])

  return new_dict

def get_currencies(text):
  curr = []
  patts = [r"(\W)([A-Za-z]{0,2}\$|USD|usd|؋|AFN|afn|ƒ|AWG|awg|AUD|aud|₼|AZN|azn|BSD|bsd|BBD|bbd|Br|BYN|byn|BZ\$|BZD|bzd|BMD|bmd|BOB|bob|\$b|BAM|bam|KM|BWP|bwp|BGN|bgn|лв|BRL|brl|R\$|BND|bnd|KHR|khr|៛|CAD|cad|KYD|kyd|CLP|clp|CNY|cny|¥|COP|cop|₡|CRC|crc|₡|HRK|hrk|kn|CUP|cup|₱|CZK|czk|Kč|DKK|dkk|kr|DOP|dop|RD\$|XCD|xcd|EGP|egp|£|SVC|svc|EUR|eur|€|FKP|fkp|FJD|fjd|GHS|ghs|¢|GIP|gip|GYD|gyd|HNL|hnl|L|HKD|hkd|HUF|huf|Ft|ISK|isk|kr|INR|inr|₹|Rs|rs|RS|IDR|idr|Rp|IRR|irr|﷼|IMP|imp|ILS|ils|₪|JMD|jmd|J\$|JPY|jpy|¥|JEP|jep|KZT|kzt|лв|KPW|kpw|₩|KRW|krw|₩|KGS|kgs|лв|LAK|lak|₭|LBP|lbp|£|LRD|lrd|MKD|mkd|ден|MYR|myr|RM|MUR|mur|₨|MXN|mxn|MNT|mnt|₮|د.إ|MZN|mzn|MT|NAD|nad|NPR|npr|₨|ANG|ang|ƒ|NZD|nzd|NIO|nio|C\$|NGN|ngn|₦|NOK|nok|kr|OMR|omr|﷼|PKR|pkr|₨|PAB|pab|PYG|pyg|Gs|PEN|pen|PHP|php|₱|PLN|pln|zł|QAR|qar|﷼|RON|ron|lei|RUB|rub|₽|SHP|shp|£|SAR|sar|﷼|RSD|rsd|Дин.|SCR|scr|₨|SGD|sgd|SBD|sbd|SOS|sos|ZAR|zar|LKR|lkr|₨|SEK|sek|kr|CHF|chf|SRD|srd|SYP|syp|£|TWD|twd|NT\$|THB|thb|฿|TTD|ttd|TT\$|TRY|try|₺|TVD|tvd|UAH|uah|₴|AED|aed|د.إ|GBP|gbp|£|UYU|uyu|\$U|UZS|uzs|лв|VEF|vef|Bs|VND|vnd|₫|YER|yer|﷼|ZWD|zwd|Z\$)( )?(([0-9]+[\,|\.]?[0-9]?)+)", "(([0-9]?[\,|\.]?[0-9])+)( )?([A-Za-z]{0,2}\$|USD|usd|؋|AFN|afn|ƒ|AWG|awg|AUD|aud|₼|AZN|azn|BSD|bsd|BBD|bbd|Br|BYN|byn|BZ\$|BZD|bzd|BMD|bmd|BOB|bob|\$b|BAM|bam|KM|BWP|bwp|BGN|bgn|лв|BRL|brl|R\$|BND|bnd|KHR|khr|៛|CAD|cad|KYD|kyd|CLP|clp|CNY|cny|¥|COP|cop|₡|CRC|crc|₡|HRK|hrk|kn|CUP|cup|₱|CZK|czk|Kč|DKK|dkk|kr|DOP|dop|RD\$|XCD|xcd|EGP|egp|£|SVC|svc|EUR|eur|€|FKP|fkp|FJD|fjd|GHS|ghs|¢|GIP|gip|GYD|gyd|HNL|hnl|L|HKD|hkd|HUF|huf|Ft|ISK|isk|kr|INR|inr|₹|Rs|rs|IDR|idr|Rp|IRR|irr|﷼|IMP|imp|ILS|ils|₪|JMD|jmd|J\$|JPY|jpy|¥|JEP|jep|KZT|kzt|лв|KPW|kpw|₩|KRW|krw|₩|KGS|kgs|лв|LAK|lak|₭|LBP|lbp|£|LRD|lrd|MKD|mkd|ден|MYR|myr|RM|MUR|mur|₨|MXN|mxn|MNT|mnt|₮|د.إ|MZN|mzn|MT|NAD|nad|NPR|npr|₨|ANG|ang|ƒ|NZD|nzd|NIO|nio|C\$|NGN|ngn|₦|NOK|nok|kr|OMR|omr|﷼|PKR|pkr|₨|PAB|pab|PYG|pyg|Gs|PEN|pen|PHP|php|₱|PLN|pln|zł|QAR|qar|﷼|RON|ron|lei|RUB|rub|₽|SHP|shp|£|SAR|sar|﷼|RSD|rsd|Дин.|SCR|scr|₨|SGD|sgd|SBD|sbd|SOS|sos|ZAR|zar|LKR|lkr|₨|SEK|sek|kr|CHF|chf|SRD|srd|SYP|syp|£|TWD|twd|NT\$|THB|thb|฿|TTD|ttd|TT\$|TRY|try|₺|TVD|tvd|UAH|uah|₴|AED|aed|د.إ|GBP|gbp|£|UYU|uyu|\$U|UZS|uzs|лв|VEF|vef|Bs|VND|vnd|₫|YER|yer|﷼|ZWD|zwd|Z\$)(\W)"]

  detected_currencies = re.finditer(patts[0], text)
  for x in detected_currencies:
    curr.append(["CURRENCY", x.group(2), x.start(), x.start() + (len(x.group(2)) + len(x.group(1)) if x.group(1) else len(x.group(2)))])
    curr.append(["AMOUNT", x.group(4), x.end() - len(x.group(4)), x.end()])

  detected_currencies = re.finditer(patts[1], text)
  for x in detected_currencies:
    curr.append(["AMOUNT", x.group(1), x.start() , x.start() + len(x.group(1)) ])
    curr.append(["CURRENCY", x.group(4), x.end() - (len(x.group(4)) + len(x.group(5)) if x.group(5) else len(x.group(4))), x.end()])

  return curr


# Import your functions here

def process_text_parallel(text):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        flair_future = executor.submit(flair_extractor, text)
        stanza_future = executor.submit(stanza_extractor, text)
        spacy_future = executor.submit(spacy_sxtractor, text)
        email_website_future = executor.submit(extract_email_website, text)
        # pincodes_future = executor.submit(pincodes, text)
        # abbrevations_future = executor.submit(abbrevations, text)
        currency_future = executor.submit(get_currencies, text)

        flair_entities = flair_future.result()
        # print(flair_entities)
        stanza_entities = stanza_future.result()
        # print(stanza_entities)
        spacy_entities = spacy_future.result()
        # print(spacy_entities)
        email_website_entities = email_website_future.result()

        # pincode_entities = pincodes_future.result()
        # abbrevation_entities = abbrevations_future.result()
        currency_entities = currency_future.result()


        # Process the results as needed
        # You can now work with the extracted entities from each function
        entities_list = []
        entities_list.extend(currency_entities)
        entities_list.extend(flair_entities)
        entities_list.extend(stanza_entities)
        entities_list.extend(spacy_entities)
        entities_list.extend(email_website_entities)
        # entities_list.extend(pincode_entities)
        # entities_list.extend(abbrevation_entities)
        
        entities_list= [e for e in entities_list if e[0]!= "MONEY"]
        return entities_list


input_sentence="Apple Inc. is headquartered in Cupertino, California. Visit their website at https://www.apple.com."


dicts_list = process_text_parallel(input_sentence)
combined_dict = list_of_lists_to_dict(dicts_list)
final_dict=merge_overlapping_entities(combined_dict)
test=filter_reduce_entity_list(final_dict)
final_dict=return_list(test)

print(final_dict)

