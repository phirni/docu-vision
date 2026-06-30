import redis
r=redis.Redis(host='localhost',port=6379,decode_response=True)
r.set('foo','bar')
print(r.get('foo'))
