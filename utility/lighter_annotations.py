import os
import csv

def create_lighter_annots(ipfile, opfile):
    ovids = []
    rows = []
    with open(ipfile, 'r') as f:
        reader = csv.DictReader(f, delimiter=' ')
        for row in reader:
            ovid = row['original_vido_id']
            labels = row['labels']
            if labels == '' or ovid in ovids:
                continue
            ovids += [ovid]
            rows += [[ovid, labels]]
            print([ovid, labels])
    header = ['video_id', 'labels']
    with open(opfile, 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    ipfile = '/ICME/AnimalKingdom/action_recognition/annotation/val.csv'
    opfile = '/ICME/AnimalKingdom/action_recognition/annotation/val_light.csv'
    create_lighter_annots(ipfile, opfile)    
