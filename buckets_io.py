import struct

def buckets_write(path, buckets_items, buckets_residue):
    f = open(path, 'wb')

    f.write(struct.pack('I', len(buckets_items)))

    for bucket, residue in zip(buckets_items, buckets_residue):
        f.write(struct.pack('I', len(bucket)))
        f.write(struct.pack('f', residue))
        if len(bucket) > 0:
            f.write(struct.pack(str(len(bucket)) + 'I', *bucket))

    f.close()
    
