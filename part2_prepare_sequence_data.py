import json
import random
import string
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import CLEANED_PATH, METADATA_PATH, DATA_DIR, OUTPUT_DIR, POS_TAGS, NER_TAGS, SEED
from utils.io_utils import read_lines, sentence_split, load_metadata, set_seed

# Expanded lexicons/gazetteers for full assignment requirements
PRONOUNS = {
    'میں','ہم','آپ','وہ','یہ','تم','اس','ان','تو','ہمیں','آپ','کو','مجھے','تمہیں','اسے','انہیں',
    'خود','اپنا','اپنی','اپنے','کوئی','کچھ','سب','سبھی','ہر','کوی','جو','جس','جن','کیا','کون',
    'کہاں','کب','کیوں','کیسے','کتنا','کتنی','کتنے','یہاں','وہاں','ادھر','ادھر','جہاں','جدھر'
}

DETERMINERS = {
    'ایک','یہ','وہ','اس','ان','تمام','ہر','کوئی','کچھ','سب','سبھی','دوسرا','دوسری','دوسرے',
    'پہلا','پہلی','پہلے','آخری','اگلا','اگلی','اگلے','پچھلا','پچھلی','پچھلے','اصل','حقیقی',
    'مخصوص','خاص','عام','مختلف','مشترکہ','الگ','جدا','متعدد','کئی','چند','بعض','تھوڑا','تھوڑی'
}

CONJUNCTIONS = {
    'اور','یا','لیکن','مگر','کہ','جب','تو','اگر','چونکہ','کیونکہ','تاکہ','جتنا','جیسا','جیسے',
    'بلکہ','بلکہ','پھر','پھر','بھی','تو','تب','جبکہ','حالانکہ','اگرچہ','تاہم','البتہ','ورنہ',
    'نہیں','تو','یعنی','مثلاً','خاص','طور','پر','عام','طور','پر','دوسری','طرف','اس','لیے'
}

POSTPOSITIONS = {
    'میں','پر','سے','کو','کا','کی','کے','تک','کے','لیے','کے','ساتھ','کے','بغیر','کے','علاوہ',
    'کے','نیچے','کے','اوپر','کے','آگے','کے','پیچھے','کے','پاس','کے','دوران','کے','بعد',
    'کے','پہلے','کے','اندر','کے','باہر','کے','درمیان','کے','ذریعے','کے','مطابق','کے','خلاف',
    'کے','حق','میں','کے','بارے','میں','کی','طرف','کی','جانب','کے','مقابلے','میں'
}

ADVERBS = {
    'بہت','کم','زیادہ','جلد','آج','کل','اب','پھر','ہمیشہ','کبھی','کبھی','نہیں','ہاں','نہ',
    'بالکل','تقریباً','لگ','بھگ','صرف','صحیح','غلط','اچھا','برا','تیز','آہستہ','دھیرے','فوری',
    'جلدی','دیر','سے','پہلے','بعد','میں','اکثر','عام','طور','پر','خاص','طور','پر','واقعی',
    'حقیقت','میں','درحقیقت','اصل','میں','یقیناً','ضرور','شاید','ممکن','ہے','یقینی','طور','پر'
}

ADJECTIVES = {
    'اچھا','بری','بڑا','چھوٹا','اہم','قومی','بین','عالمی','نیا','پرانا','جوان','بوڑھا','خوبصورت',
    'بدصورت','امیر','غریب','طاقتور','کمزور','ذہین','احمق','محنتی','سست','ایماندار','بے','ایمان',
    'مہربان','سخت','نرم','گرم','ٹھنڈا','تیز','سست','اونچا','نیچا','لمبا','چوڑا','تنگ','موٹا',
    'پتلا','صاف','گندا','روشن','اندھیرا','خوش','غمگین','پرامن','خطرناک','محفوظ','آزاد','قید'
}

VERB_SUFFIXES = (
    'نا','تا','تی','تے','گا','گی','گے','رہا','رہی','رہے','چکا','چکی','چکے','سکتا','سکتی','سکتے',
    'پایا','پائی','پائے','دیا','دی','دیے','لیا','لی','لیے','کیا','کی','کیے','ہوا','ہوئی','ہوئے',
    'جانا','آنا','کرنا','ہونا','دینا','لینا','کہنا','سننا','دیکھنا','سمجھنا','پڑھنا','لکھنا'
)

# Expanded NER Gazetteers
ORG_GAZ = {
    'اقوام','متحدہ','پی','سی','بی','سپریم','کورٹ','پارلیمنٹ','حکومت','وزارت','محکمہ','کمیشن',
    'بورڈ','کونسل','کمیٹی','فیڈریشن','ایسوسی','ایشن','یونین','پارٹی','تحریک','جماعت','لیگ',
    'پیپلز','پارٹی','تحریک','انصاف','مسلم','لیگ','جمعیت','علماء','اسلام','جماعت','اسلامی',
    'عوامی','نیشنل','پارٹی','بلوچستان','نیشنل','پارٹی','متحدہ','قومی','موومنٹ','پاکستان',
    'تحریک','طالبان','القاعدہ','داعش','نیٹو','یورپی','یونین','عرب','لیگ','آسیان','سارک',
    'ورلڈ','بینک','آئی','ایم','ایف','یونیسیف','ڈبلیو','ایچ','او','یونیسکو','فیفا','آئی','سی','سی'
}

LOC_GAZ = {
    'پاکستان','لاہور','کراچی','اسلام','آباد','پشاور','کوئٹہ','سندھ','پنجاب','بلوچستان','خیبر',
    'پختونخوا','گلگت','بلتستان','آزاد','کشمیر','فیصل','آباد','راولپنڈی','ملتان','گجرانوالہ',
    'حیدر','آباد','سکھر','بہاولپور','سرگودھا','شیخوپورہ','جھنگ','گجرات','کسور','اوکاڑہ',
    'ہندوستان','بھارت','چین','افغانستان','ایران','ترکی','سعودی','عرب','امریکہ','برطانیہ',
    'فرانس','جرمنی','روس','جاپان','آسٹریلیا','کینیڈا','برازیل','مصر','نائیجیریا','جنوبی',
    'افریقہ','انڈونیشیا','ملائیشیا','تھائی','لینڈ','ویتنام','کوریا','اسرائیل','فلسطین','عراق'
}

PER_GAZ = {
    'عمران','خان','شہباز','شریف','نواز','شریف','بلاول','بھٹو','زرداری','بابر','اعظم','رضوان',
    'افریدی','محمد','علی','احمد','حسن','حسین','فاطمہ','عائشہ','خدیجہ','زینب','مریم','صفیہ',
    'عبداللہ','عبدالرحمن','عبدالرحیم','عبدالکریم','عبدالحمید','عبدالمجید','عبدالغفور',
    'عبدالستار','عبدالقادر','عبدالرشید','عبدالحق','عبدالوہاب','عبدالصمد','عبدالحلیم',
    'قائد','اعظم','علامہ','اقبال','لیاقت','علی','خان','فاطمہ','جناح','بے','نظیر','بھٹو',
    'ذوالفقار','علی','بھٹو','ایوب','خان','یحیٰی','خان','ضیاء','الحق','پرویز','مشرف'
}

MISC_GAZ = {
    'رمضان','عید','الفطر','عید','الاضحیٰ','محرم','صفر','ربیع','الاول','ربیع','الثانی','جمادی',
    'الاول','جمادی','الثانی','رجب','شعبان','شوال','ذوالقعدہ','ذوالحجہ','ورلڈ','کپ','ایشیا',
    'کپ','چیمپئنز','ٹرافی','ٹی','ٹوئنٹی','ون','ڈے','ٹیسٹ','میچ','اولمپکس','کامن','ویلتھ',
    'گیمز','فیفا','ورلڈ','کپ','یورو','کپ','کوپا','امریکہ','افریقن','کپ','آف','نیشنز',
    'نوبل','انعام','آسکر','ایمی','گریمی','گولڈن','گلوب','بافٹا','کان','فلم','فیسٹیول',
    'برلن','فلم','فیسٹیول','وینس','فلم','فیسٹیول','سن','ڈانس','فلم','فیسٹیول'
}

PUNCS = set(string.punctuation) | {'،', '۔', '؛', '؟', '!', ':'}


def infer_topic(meta_item):
    if isinstance(meta_item, dict):
        for key in ['topic', 'category', 'label', 'class']:
            if key in meta_item:
                return str(meta_item[key])
    return 'unknown'


def pos_tag_token(tok: str) -> str:
    # Handle punctuation first
    if tok in PUNCS:
        return 'PUNC'
    
    # Handle numbers
    if tok.isdigit() or any(c.isdigit() for c in tok):
        return 'NUM'
    
    # Check specific word categories
    if tok in PRONOUNS:
        return 'PRON'
    if tok in DETERMINERS:
        return 'DET'
    if tok in CONJUNCTIONS:
        return 'CONJ'
    if tok in POSTPOSITIONS:
        return 'POST'
    if tok in ADVERBS:
        return 'ADV'
    if tok in ADJECTIVES:
        return 'ADJ'
    
    # Check verb patterns (more comprehensive)
    if any(tok.endswith(suffix) for suffix in VERB_SUFFIXES):
        return 'VERB'
    
    # Common verb patterns in Urdu
    if tok.endswith(('یں', 'ے', 'ا', 'و')) and len(tok) > 2:
        # Could be verb forms
        root = tok[:-1] if tok.endswith(('یں', 'ے', 'ا', 'و')) else tok
        if any(root.endswith(suffix) for suffix in ('کر', 'ہو', 'جا', 'آ', 'دے', 'لے')):
            return 'VERB'
    
    # Handle single character words
    if len(tok) == 1:
        if tok in {'و', 'یا'}:  # Common conjunctions
            return 'CONJ'
        elif tok in PUNCS:
            return 'PUNC'
        else:
            return 'UNK'
    
    # Default to NOUN for longer words that don't match other categories
    return 'NOUN'


def ner_tags(tokens):
    tags = ['O'] * len(tokens)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        
        # Check for multi-word entities first
        if i < len(tokens) - 1:
            two_word = f"{tok}_{tokens[i+1]}"
            if two_word in PER_GAZ or two_word in LOC_GAZ or two_word in ORG_GAZ or two_word in MISC_GAZ:
                if two_word in PER_GAZ:
                    tags[i] = 'B-PER'
                    tags[i+1] = 'I-PER'
                elif two_word in LOC_GAZ:
                    tags[i] = 'B-LOC'
                    tags[i+1] = 'I-LOC'
                elif two_word in ORG_GAZ:
                    tags[i] = 'B-ORG'
                    tags[i+1] = 'I-ORG'
                elif two_word in MISC_GAZ:
                    tags[i] = 'B-MISC'
                    tags[i+1] = 'I-MISC'
                i += 2
                continue
        
        # Check for three-word entities
        if i < len(tokens) - 2:
            three_word = f"{tok}_{tokens[i+1]}_{tokens[i+2]}"
            if three_word in PER_GAZ or three_word in LOC_GAZ or three_word in ORG_GAZ or three_word in MISC_GAZ:
                if three_word in PER_GAZ:
                    tags[i] = 'B-PER'
                    tags[i+1] = 'I-PER'
                    tags[i+2] = 'I-PER'
                elif three_word in LOC_GAZ:
                    tags[i] = 'B-LOC'
                    tags[i+1] = 'I-LOC'
                    tags[i+2] = 'I-LOC'
                elif three_word in ORG_GAZ:
                    tags[i] = 'B-ORG'
                    tags[i+1] = 'I-ORG'
                    tags[i+2] = 'I-ORG'
                elif three_word in MISC_GAZ:
                    tags[i] = 'B-MISC'
                    tags[i+1] = 'I-MISC'
                    tags[i+2] = 'I-MISC'
                i += 3
                continue
        
        # Single word entities
        if tok in PER_GAZ:
            tags[i] = 'B-PER'
        elif tok in LOC_GAZ:
            tags[i] = 'B-LOC'
        elif tok in ORG_GAZ:
            tags[i] = 'B-ORG'
        elif tok in MISC_GAZ:
            tags[i] = 'B-MISC'
        
        i += 1
    return tags


def write_conll(samples, path: Path, task: str):
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            for tok, label in zip(sample['tokens'], sample[task]):
                f.write(f'{tok}\t{label}\n')
            f.write('\n')


def label_distribution(samples, key):
    c = Counter()
    for s in samples:
        c.update(s[key])
    return dict(c)


def main():
    set_seed()
    docs = read_lines(CLEANED_PATH)
    meta = load_metadata(METADATA_PATH) if METADATA_PATH.exists() else [{'topic': 'unknown'} for _ in docs]

    sentences_by_topic = defaultdict(list)
    for doc, m in zip(docs, meta):
        topic = infer_topic(m)
        for sent in sentence_split(doc):
            if len(sent) >= 3:
                sentences_by_topic[topic].append(sent)

    topics = list(sentences_by_topic.keys())
    chosen_topics = topics[:3] if len(topics) >= 3 else topics
    samples = []
    for topic in chosen_topics:
        picks = random.sample(sentences_by_topic[topic], min(100, len(sentences_by_topic[topic])))
        for sent in picks:
            samples.append({
                'tokens': sent,
                'pos': [pos_tag_token(t) for t in sent],
                'ner': ner_tags(sent),
                'topic': topic,
            })

    remaining = 500 - len(samples)
    pool = [s for topic_sents in sentences_by_topic.values() for s in topic_sents]
    for sent in random.sample(pool, min(remaining, len(pool))):
        samples.append({
            'tokens': sent,
            'pos': [pos_tag_token(t) for t in sent],
            'ner': ner_tags(sent),
            'topic': 'mixed',
        })

    topic_labels = [s['topic'] for s in samples]
    train_samples, temp_samples = train_test_split(samples, test_size=0.30, random_state=SEED, stratify=topic_labels)
    temp_labels = [s['topic'] for s in temp_samples]
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.50, random_state=SEED, stratify=temp_labels)

    write_conll(train_samples, DATA_DIR / 'pos_train.conll', 'pos')
    write_conll(val_samples, DATA_DIR / 'pos_val.conll', 'pos')
    write_conll(test_samples, DATA_DIR / 'pos_test.conll', 'pos')
    write_conll(train_samples, DATA_DIR / 'ner_train.conll', 'ner')
    write_conll(val_samples, DATA_DIR / 'ner_val.conll', 'ner')
    write_conll(test_samples, DATA_DIR / 'ner_test.conll', 'ner')

    with open(OUTPUT_DIR / 'sequence_label_distributions.json', 'w', encoding='utf-8') as f:
        json.dump({
            'pos_train': label_distribution(train_samples, 'pos'),
            'pos_val': label_distribution(val_samples, 'pos'),
            'pos_test': label_distribution(test_samples, 'pos'),
            'ner_train': label_distribution(train_samples, 'ner'),
            'ner_val': label_distribution(val_samples, 'ner'),
            'ner_test': label_distribution(test_samples, 'ner'),
        }, f, ensure_ascii=False, indent=2)
    print('Saved POS/NER conll splits and label distributions.')


if __name__ == '__main__':
    main()
