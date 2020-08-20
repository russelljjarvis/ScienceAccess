from science_access.crawl import collect_pubs
from science_access.get_bmark_corpus import get_bmarks
scholar_link=str('https://duckduckgo.com/?q=%21scholar')+str("ninkum")+str("&va=z&t=hc")
follow_links = collect_pubs(scholar_link)
print(follow_links)
urlDats = get_bmarks()
for ur in urlDats:
    assert ur['standard'] <90

