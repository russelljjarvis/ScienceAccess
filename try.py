from crossref_commons.iteration import iterate_publications_as_json
filter = {'type': 'journal-article'}
queries = {'query.author': 'McGurrin'}
import os
for p in iterate_publications_as_json(max_results=189, filter=filter, queries=queries):   
    if 'abstract' in p.keys(): 
        print(p.keys()) 
        temp='wget https://unpaywall.org/'+str(p['DOI']) 
        print(temp) 
        os.system(temp) 
