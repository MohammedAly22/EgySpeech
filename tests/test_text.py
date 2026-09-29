from egyspeech.text import (
    TagSet,
    arabic_ratio,
    cer,
    clean_llm_output,
    clean_tags,
    has_repetition_loop,
    is_code_switched,
    latin_words,
    normalize_for_compare,
    strip_tags,
)

TAGS = TagSet(events=["laughs", "sighs", "breath", "clears_throat"], spans=["whispering"], styles=["excited"])


def test_code_switching_detection():
    assert is_code_switched("الـ meeting بتاعنا كان very productive")
    assert latin_words("الـ meeting بتاعنا كان very productive") == ["meeting", "very", "productive"]
    assert not is_code_switched("النهارده الجو حلو قوي")
    assert arabic_ratio("hello world") == 0.0


def test_tags_validated_and_canonical():
    text, found = clean_tags("[Excited] انا مبسوط جدا [LAUGHS] [dances] [clears throat] ماشي", TAGS)
    assert text == "[excited] انا مبسوط جدا [laughs] [clears_throat] ماشي"
    assert found == ["excited", "laughs", "clears_throat"]


def test_unclosed_span_is_closed_and_stray_close_dropped():
    text, found = clean_tags("[whispering] سر بيني وبينك [/sighs] خلاص", TAGS)
    assert text == "[whispering] سر بيني وبينك خلاص [/whispering]"
    assert found == ["whispering"]


def test_strip_tags():
    assert strip_tags("[excited] اهلا [laughs] بيك [whispering] يا [/whispering] باشا") == "اهلا بيك يا باشا"


def test_clean_llm_output():
    assert clean_llm_output("```\nTranscript: اهلا بيك\n```") == "اهلا بيك"
    assert clean_llm_output('"اهلا"') == "اهلا"


def test_compare_normalization_and_cer():
    a = normalize_for_compare("أنا رايح المدرسة، [laughs] إزيك؟")
    b = normalize_for_compare("انا رايح المدرسه ازيك")
    assert a == b
    assert cer(a, b) == 0.0
    assert 0 < cer("abcd", "abxd") <= 0.25


def test_repetition_loop():
    assert has_repetition_loop("يعني يعني يعني يعني يعني يعني يعني")
    assert has_repetition_loop("انا قلت " * 8)
    assert not has_repetition_loop("انا قلت له كده وهو رد عليا بسرعة")
