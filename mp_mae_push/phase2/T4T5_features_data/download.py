import urllib.request, ssl, hashlib, json, time, sys, os
url=sys.argv[1]; out=sys.argv[2]
ctx=ssl.create_default_context()
h=hashlib.sha256(); m=hashlib.md5(); n=0; t0=time.time()
req=urllib.request.Request(url)
with urllib.request.urlopen(req,timeout=120,context=ctx) as r, open(out+'.part','wb') as f:
    final=r.geturl()
    while True:
        b=r.read(1<<20)
        if not b: break
        f.write(b); h.update(b); m.update(b); n+=len(b)
os.replace(out+'.part',out)
json.dump(dict(url=url,final_url_host=final.split('/')[2],bytes=n,sha256=h.hexdigest(),md5=m.hexdigest(),seconds=time.time()-t0,
  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),open(out+'.meta.json','w'),indent=1)
print('DONE',n)
