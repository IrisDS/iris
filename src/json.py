import json


if __name__ == "__main__":
	myfile="manfest.json"
	data = open(myfile,'r')
	f=json.load(data)

	#head wrapper
	t=0
	fatdump={}
	scrs={}
	for i in f["nodes"]:
		ips=f["nodes"][t]['ip']
		imgs=f["nodes"][t]["image"]
		fatdump[ips]=t
		fatdump[imgs]=t
		#score wrapper
		h=0
		for j in f["nodes"]:
			scrs[t,h]=score[t,h]
			#scrs[t,h]=t+h
			fatdump["scr"]=scrs
			h+=1
		t+=1

print fatdump
#print json.dumps(fatdump, sort_keys=True, indent=4,separators=(',',': '))
