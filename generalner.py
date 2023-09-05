from flair.data import Sentence
from flair.models import SequenceTagger
import stanza
import spacy

import concurrent.futures
import re
import string
import time


tagger = SequenceTagger.load("flair/ner-english-ontonotes")
nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,ner')
nlp_spacy = spacy.load("en_core_web_lg")

def flair_extractor(text):
  # tagger = SequenceTagger.load("flair/ner-english-ontonotes")
    sentence_flair = Sentence(text)
    tagger.predict(sentence_flair)
    entities_flair = sentence_flair.get_spans('ner')
    entity_list_flair = {}
    for entity in entities_flair:
        entity_list_flair[entity.text] = entity.tag
    return entity_list_flair

def stanza_extractor(text):
    # nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,ner')
    doc_stanza = nlp_stanza(text)
    entities_stanza = {}
    for sent in doc_stanza.sentences:
        for entity in sent.ents:
            entities_stanza[entity.text] = entity.type
    return entities_stanza



def spacy_sxtractor(text):
    # nlp_spacy = spacy.load("en_core_web_lg")
    doc_spacy = nlp_spacy(text)
    entities_spacy = {}
    for ent in doc_spacy.ents:
        entities_spacy[ent.text] = ent.label_
    return entities_spacy

def extract_email_website(text):
  l={}
  email_regex= r"([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)"
  email_list=re.findall(email_regex, text)
  web ="(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?([a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?"
  web_list=[x[1] for x in re.findall(web, text)]
  for i in email_list:
    l[i]='EMAIL'
  for i in web_list:
    l[i]= 'WEB_LINK'
  return l

def pincodes(text):
  pincode = {"IN" : {"pin" : "\\b([1-9]{1}[0-9]{2}\\s{0,1}\\-{0,1}[0-9]{3})\\b", "len" : 6, "ll" : [744101,507130,790001,781001,800001,140119,490001,396193,362520,110001,403001,360001,121001,171001,180001,813208,560001,670001,682551,450001,400001,795001,783123,796001,797001,751001,533464,140001,301001,737101,600001,500001,799001,201001,244712,700001], "up" : [744304,535594,792131,788931,855117,160102,497778,396240,396220,110097,403806,396590,136156,177601,194404,835325,591346,695615,682559,488448,445402,795159,794115,796901,798627,770076,673310,160104,345034,737139,643253,509412,799290,285223,263680,743711], "rules" : ["10", "29", "35", "54", "55", "65", "66", "86", "87", "88", "89"]},
            "USA": {"pin" : "\\b[0-9]{5}(?:-[0-9]{4})?\\b", "len" : 5, "ll" : [501,1001,2801,3031,3901,5001,6001,7001,15001,19701,20101,20588,24701,27006,29001,30002,32003,35004,37010,38601,40003,43001,46001,48001,50001,53001,55001,57001,58001,59001,60001,63001,66002,68001,70001,71601,73001,73301,80001,82001,83201,84001,85001,87001,88901,90001,96701,97001,98001,99501], "up" : [14925,5544,2940,3897,4992,5907,6928,8989,19640,19980,24658,21930,26886,28909,29945,39901,34997,36925,38589,39776,42788,45999,47997,49971,52809,54990,56763,57799,58856,59937,62999,65899,67954,69367,71497,72959,74966,88595,81658,83414,83877,84791,86556,88439,89883,96162,96898,97920,99403,99950]},
            "GE" : {"pin" : "\\b[0-9]{5}\\b", "len" : 6, "ll" : [1067,3042,6108,7318,10115,17033,20038,21217,22844,27568,32049,34117,54290,66041,68131,80331,8056,14461,38820,98527,22041,26121,28195,40196,60306,66849,88212,49074,58084,64283], "up" : [4889,3253,6928,7989,14532,19417,21149,21789,25999,27580,33829,37299,57648,66839,79879,97909,9669,17326,39649,99998,22769,38729,28779,54585,63699,67829,89198,49849,59969,65936]},
            "SN" : {"pin" : "\\b([0-8]{1}[0-9]{5})\\b" , "len" : 6,  "ll" : [10000],"up" : [829999]},
            "UK" : {"pin" : "\\b(([A-Z]{1,2}\\d[A-Z\\d]?|ASCN|STHL|TDCU|BBND|[BFS]IQQ|PCRN|TKCA) ?\\d[A-Z]{2}|BFPO ?\\d{1,4}|(KY\\d|MSR|VG|AI)[ -]?\\d{4}|[A-Z]{2} ?\\d{2}|GE ?CX|GIR ?0A{2}|SAN ?TA1)\\b"}
          }
  l = {}
  for country, sub in pincode.items():
    t = re.finditer(sub["pin"], text)

    if "ll" not in sub:
      for x in t:
        l[x.group()] = ["PINCODE-" + country]
        # l.append([ x.start(), x.end(), x.group(), "pin-" + country])

    else:
      for x in t:
        z = int(x.group().replace("-","").replace(" ","")[0:sub["len"]])
        for i in range(len(sub["ll"])):
          if z>= sub["ll"][i] and z<= sub["up"][i]:
            if x.group() in l.keys():
              l[x.group()].append("PINCODE-" + country)
              break
            else:
              l[x.group()] = ["PINCODE-" + country]
              break
            # l.append([ x.start(), x.end(), x.group(), "pin-" + country])

  return l

def abbrevations(text):
  i = 0
  l = {}
  for t in text.split(" "):
    score = 0
    score += 2 if len(t.replace(".", "")) < 6 else 0
    score += 3 if t.isupper() else 0
    score += 2 if re.search(r"(([A-Z]\.|\-){1,4}[A-Z]|\W)", t) else 0 #U.S.A
    score += 3 if re.search(r"(\([A-Z]{2,5}\))", t) else 0 # (WHO)
    score, t  = (score + 5, re.search(r"\bDr.|dr.|Mrs.|Mr.|mrs.|mr.\b", t).group())if re.search(r"\bDr.|dr.|Mrs.|Mr.|mrs.|mr.\b", t) else (score + 0, t)#Dr.
    if score > 4:
      l[t.translate(str.maketrans('', '', string.punctuation))] = "ABBREVATION"
    i+= len(t) + 1
  return l

def get_currencies(text):
  curr_dict = {}
  patt = r"\W[A-Za-z]{0,3}\$\W|\WUSD\W|\Wusd\W|\W؋\W|\WAFN\W|\Wafn\W|\Wƒ\W|\WAWG\W|\Wawg\W|\W₼\W|\WAZN\W|\Wazn\W|\WBSD\W|\Wbsd\W|\WBBD\W|\Wbbd\W|\WBr\W|\WBYN\W|\Wbyn\W|\WBZ\$\W|\WBZD\W|\Wbzd\W|\WBMD\W|\Wbmd\W|\WBOB\W|\Wbob\W|\W\$b\W|\WBAM\W|\Wbam\W|\WKM\W|\WBWP\W|\Wbwp\W|\WP\W|\WBGN\W|\Wbgn\W|\Wлв\W|\WBRL\W|\Wbrl\W|\WR\$\W|\WBND\W|\Wbnd\W|\WKHR\W|\Wkhr\W|\W៛\W|\WCAD\W|\Wcad\W|\WKYD\W|\Wkyd\W|\WCLP\W|\Wclp\W|\WCNY\W|\Wcny\W|\W¥\W|\WCOP\W|\Wcop\W|\W₡\W|\WCRC\W|\Wcrc\W|\W₡\W|\WHRK\W|\Whrk\W|\Wkn\W|\WCUP\W|\Wcup\W|\W₱\W|\WCZK\W|\Wczk\W|\WKč\W|\WDKK\W|\Wdkk\W|\Wkr\W|\WDOP\W|\Wdop\W|\WRD\$\W|\WXCD\W|\Wxcd\W|\WEGP\W|\Wegp\W|\W£\W|\WSVC\W|\Wsvc\W|\WEUR\W|\Weur\W|\W€\W|\WFKP\W|\Wfkp\W|\W£\W|\WFJD\W|\Wfjd\W|\WGHS\W|\Wghs\W|\W¢\W|\WGIP\W|\Wgip\W|\W£\W|\WGYD\W|\Wgyd\W|\WHNL\W|\Whnl\W|\WL\W|\WHKD\W|\Whkd\W|\W\$|WHUF\W|\Whuf\W|\WFt\W|\WISK\W|\Wisk\W|\Wkr\W|\WINR\W|\Winr\W|\W₹\W|\WIDR\W|\Widr\W|\WRp\W|\WIRR\W|\Wirr\W|\W﷼\W|\WIMP\W|\Wimp\W|\W£\W|\WILS\W|\Wils\W|\W₪\W|\WJMD\W|\Wjmd\W|\WJ\$\W|\WJPY\W|\Wjpy\W|\W¥\W|\WJEP\W|\Wjep\W|\W£\W|\WKZT\W|\Wkzt\W|\Wлв\W|\WKPW\W|\Wkpw\W|\W₩\W|\WKRW\W|\Wkrw\W|\W₩\W|\WKGS\W|\Wkgs\W|\Wлв\W|\WLAK\W|\Wlak\W|\W₭\W|\WLBP\W|\Wlbp\W|\W£\W|\WLRD\W|\Wlrd\W|\W\$\W|\WMKD\W|\Wmkd\W|\Wден\W|\WMYR\W|\Wmyr\W|\WRM\W|\WMUR\W|\Wmur\W|\W₨\W|\WMXN\W|\Wmxn\W|\W\$\W|\WMNT\W|\Wmnt\W|\W₮\W|\Wد.إ\W|\WMZN\W|\Wmzn\W|\WMT\W|\WNAD\W|\Wnad\W|\WNPR\W|\Wnpr\W|\W₨\W|\WANG\W|\Wang\W|\Wƒ\W|\WNZD\W|\Wnzd\W|\WNIO\W|\Wnio\W|\WC\$\W|\WNGN\W|\Wngn\W|\W₦\W|\WNOK\W|\Wnok\W|\Wkr\W|\WOMR\W|\Womr\W|\W﷼\W|\WPKR\W|\Wpkr\W|\W₨\W|\WPAB\W|\Wpab\W|\WB/\.\W|\WPYG\W|\Wpyg\W|\WGs\W|\WPEN\W|\Wpen\W|\WS/\.\W|\WPHP\W|\Wphp\W|\W₱\W|\WPLN\W|\Wpln\W|\Wzł\W|\WQAR\W|\Wqar\W|\W﷼\W|\WRON\W|\Wron\W|\Wlei\W|\WRUB\W|\Wrub\W|\W₽\W|\WSHP\W|\Wshp\W|\W£\W|\WSAR\W|\Wsar\W|\W﷼\W|\WRSD\W|\Wrsd\W|\WДин\.\W|\WSCR\W|\Wscr\W|\W₨\W|\WSGD\W|\Wsgd\W|\WSBD\W|\Wsbd\W|\WSOS\W|\Wsos\W|\WS\W|\WZAR\W|\Wzar\W|\WR\W|\WLKR\W|\Wlkr\W|\W₨\W|\WSEK\W|\Wsek\W|\Wkr\W|\WCHF\W|\Wchf\W|\WCHF\W|\WSRD\W|\Wsrd\W|\WSYP\W|\Wsyp\W|\W£\W|\WTWD\W|\Wtwd\W|\WNT\$\W|\WTHB\W|\Wthb\W|\W฿\W|\WTTD\W|\Wttd\W|\WTT\$\W|\WTRY\W|\Wtry\W|\W₺\W|\WTVD\W|\Wtvd\W|\W\$\W|\WUAH\
"
  detected_currencies = re.finditer(patt, text)
  curr_dict = {curr.group() : "CURRENCY" for curr in detected_currencies}
  return curr_dict

def combine_unique_keys(dicts_list):
    combined_dict = {}

    for d in dicts_list:
        for key, value in d.items():
            if isinstance(value, list):
                for item in value:
                    if key in combined_dict:
                        if item not in combined_dict[key]:
                            combined_dict[key].append(item)
                    else:
                        combined_dict[key] = [item]
            else:
                if key in combined_dict:
                    if value not in combined_dict[key]:
                        combined_dict[key].append(value)
                else:
                    combined_dict[key] = [value]

    return combined_dict




# Import your functions here

def process_text_parallel(text):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        flair_future = executor.submit(flair_extractor, text)
        stanza_future = executor.submit(stanza_extractor, text)
        spacy_future = executor.submit(spacy_sxtractor, text)
        email_website_future = executor.submit(extract_email_website, text)
        pincodes_future = executor.submit(pincodes, text)
        abbrevations_future = executor.submit(abbrevations, text)
        currency_future = executor.submit(get_currencies, text)

        flair_entities = flair_future.result()
        stanza_entities = stanza_future.result()
        spacy_entities = spacy_future.result()
        email_website_entities = email_website_future.result()
        pincode_entities = pincodes_future.result()
        abbrevation_entities = abbrevations_future.result()
        currency_entities = currency_future.result()


        # Process the results as needed
        # You can now work with the extracted entities from each function
        entities_list = []
        entities_list.append(flair_entities)
        entities_list.append(stanza_entities)
        entities_list.append(spacy_entities)
        entities_list.append(email_website_entities)
        entities_list.append(pincode_entities)
        entities_list.append(abbrevation_entities)
        entities_list.append(currency_entities)

        return entities_list


def convert_dict(entities):
    end_dict={}

    for keys,values_list in entities.items():
        for value in values_list:
           if value not in end_dict.keys():
            end_dict[value]=[]
    
    for keys,values_list in entities.items():
        for value in values_list:
           end_dict[value].append(keys)

    return end_dict
    

# Replace the following text with the input text you want to process
input_text = f'''
On a sunny morning in August 2023, Dr. Emily Johnson, a renowned astrophysicist, arrived at the headquarters of
the International Space Exploration Consortium (ISEC), located in Houston, Texas, USA. She had been invited to
'''

dicts_list = process_text_parallel(input_text)
combined_dict = combine_unique_keys(dicts_list)
dict=convert_dict(combined_dict)
print(dict)