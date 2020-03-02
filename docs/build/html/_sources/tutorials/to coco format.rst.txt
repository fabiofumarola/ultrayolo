.. code:: ipython3

    import json
    from pathlib import Path

.. code:: ipython3

    base_path = Path('../tests/data/')

.. code:: ipython3

    !ls {base_path}


.. parsed-literal::

    __init__.py        classes.txt        [1m[36mother_images[m[m
    [1m[36mannotations[m[m        [1m[36mimages[m[m             train.yml
    annotations.txt    manifest.txt       yolov3_anchors.txt


.. code:: ipython3

    doc = {
        'info': {
            "description": "COCO test Dataset",
            "url": "",
            "version": "1.0",
            "year": 2020,
            "contributor": "Fabio",
            "date_created": "2020/01/27"
        },
        'licenses': {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
        },
        'images': [],
        'annotations': [],
        'categories': []
    }

.. code:: ipython3

    doc




.. parsed-literal::

    {'info': {'description': 'COCO test Dataset',
      'url': '',
      'version': '1.0',
      'year': 2020,
      'contributor': 'Fabio',
      'date_created': '2020/01/27'},
     'licenses': {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
      'id': 1,
      'name': 'Attribution-NonCommercial-ShareAlike License'},
     'images': [],
     'annotations': [],
     'categories': []}



.. code:: ipython3

    from imageio import imread
    from datetime import datetime

.. code:: ipython3

    images = []
    for i, img_path in enumerate((base_path / 'images').glob('*.jpg')):
        img = imread(img_path)
        img_doc = {
            "license": 1,
            'file_name': img_path.name,
            'coco_url': '',
            "height": img.shape[0],
            "width": img.shape[1],
            "date_captured": datetime.now().strftime('%Y-%m-%d %H:%m:%S'),
            "flickr_url": "",
            "id": i
            
        }
        images.append(img_doc) 
        
    doc['images'] = images

.. code:: ipython3

    doc




.. parsed-literal::

    {'info': {'description': 'COCO test Dataset',
      'url': '',
      'version': '1.0',
      'year': 2020,
      'contributor': 'Fabio',
      'date_created': '2020/01/27'},
     'licenses': {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
      'id': 1,
      'name': 'Attribution-NonCommercial-ShareAlike License'},
     'images': [{'license': 1,
       'file_name': 'AK65JZA-FORD-CAR_1.jpg',
       'coco_url': '',
       'height': 250,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 0},
      {'license': 1,
       'file_name': 'AK65JZA-FORD-CAR_0.jpg',
       'coco_url': '',
       'height': 251,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 1},
      {'license': 1,
       'file_name': 'car.jpg',
       'coco_url': '',
       'height': 1080,
       'width': 1920,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 2},
      {'license': 1,
       'file_name': 'BG55HWH-CITROEN_0.jpg',
       'coco_url': '',
       'height': 502,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 3}],
     'annotations': [],
     'categories': []}



.. code:: ipython3

    with open(base_path /'classes.txt', 'r') as fp:
        classes = [v.strip() for v in fp.readlines()]
        
    categories = []
    for i, cat in enumerate(classes):
        cat_doc = {
            "supercategory": "thing",
            "id": i,
            "name": cat
        }
        categories.append(cat_doc)
    doc['categories'] = categories

.. code:: ipython3

    doc




.. parsed-literal::

    {'info': {'description': 'COCO test Dataset',
      'url': '',
      'version': '1.0',
      'year': 2020,
      'contributor': 'Fabio',
      'date_created': '2020/01/27'},
     'licenses': {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
      'id': 1,
      'name': 'Attribution-NonCommercial-ShareAlike License'},
     'images': [{'license': 1,
       'file_name': 'AK65JZA-FORD-CAR_1.jpg',
       'coco_url': '',
       'height': 250,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 0},
      {'license': 1,
       'file_name': 'AK65JZA-FORD-CAR_0.jpg',
       'coco_url': '',
       'height': 251,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 1},
      {'license': 1,
       'file_name': 'car.jpg',
       'coco_url': '',
       'height': 1080,
       'width': 1920,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 2},
      {'license': 1,
       'file_name': 'BG55HWH-CITROEN_0.jpg',
       'coco_url': '',
       'height': 502,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 3}],
     'annotations': [],
     'categories': [{'supercategory': 'thing', 'id': 0, 'name': 'book'},
      {'supercategory': 'thing', 'id': 1, 'name': 'bus'},
      {'supercategory': 'thing', 'id': 2, 'name': 'car'},
      {'supercategory': 'thing', 'id': 3, 'name': 'motorcycle'},
      {'supercategory': 'thing', 'id': 4, 'name': 'vehicle registration plate'}]}



.. code:: ipython3

    c = 0
    annotations = []
    for i, ann_file in enumerate((base_path / 'annotations').glob('*.txt')):
        ann_list = [v.split(',') for v in ann_file.read_text().splitlines()]
        for ann in ann_list:
            ann_doc = {
                'segmentation': [],
                'area': 0,
                'iscrowd': 0,
                'image_id': i,
                'bbox': [float(v) for v in ann[:4]],
                'category_id': int(ann[4]),
                'id': c
            }
            annotations.append(ann_doc)
            c += 1
    doc['annotations'] = annotations        

.. code:: ipython3

    doc




.. parsed-literal::

    {'info': {'description': 'COCO test Dataset',
      'url': '',
      'version': '1.0',
      'year': 2020,
      'contributor': 'Fabio',
      'date_created': '2020/01/27'},
     'licenses': {'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
      'id': 1,
      'name': 'Attribution-NonCommercial-ShareAlike License'},
     'images': [{'license': 1,
       'file_name': 'AK65JZA-FORD-CAR_1.jpg',
       'coco_url': '',
       'height': 250,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 0},
      {'license': 1,
       'file_name': 'AK65JZA-FORD-CAR_0.jpg',
       'coco_url': '',
       'height': 251,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 1},
      {'license': 1,
       'file_name': 'car.jpg',
       'coco_url': '',
       'height': 1080,
       'width': 1920,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 2},
      {'license': 1,
       'file_name': 'BG55HWH-CITROEN_0.jpg',
       'coco_url': '',
       'height': 502,
       'width': 502,
       'date_captured': '2020-01-27 19:01:01',
       'flickr_url': '',
       'id': 3}],
     'annotations': [{'segmentation': [],
       'area': 0,
       'iscrowd': 0,
       'image_id': 0,
       'bbox': [0.0, 22.0, 520.0, 258.0],
       'category_id': 2,
       'id': 0},
      {'segmentation': [],
       'area': 0,
       'iscrowd': 0,
       'image_id': 0,
       'bbox': [0.0, 0.0, 442.0, 500.0],
       'category_id': 2,
       'id': 1},
      {'segmentation': [],
       'area': 0,
       'iscrowd': 0,
       'image_id': 1,
       'bbox': [0.0, 8.0, 517.0, 224.0],
       'category_id': 2,
       'id': 2},
      {'segmentation': [],
       'area': 0,
       'iscrowd': 0,
       'image_id': 2,
       'bbox': [87.0, 82.0, 536.0, 247.0],
       'category_id': 2,
       'id': 3},
      {'segmentation': [],
       'area': 0,
       'iscrowd': 0,
       'image_id': 3,
       'bbox': [0.0, 0.0, 492.0, 500.0],
       'category_id': 2,
       'id': 4},
      {'segmentation': [],
       'area': 0,
       'iscrowd': 0,
       'image_id': 3,
       'bbox': [0.0, 0.0, 442.0, 500.0],
       'category_id': 2,
       'id': 5}],
     'categories': [{'supercategory': 'thing', 'id': 0, 'name': 'book'},
      {'supercategory': 'thing', 'id': 1, 'name': 'bus'},
      {'supercategory': 'thing', 'id': 2, 'name': 'car'},
      {'supercategory': 'thing', 'id': 3, 'name': 'motorcycle'},
      {'supercategory': 'thing', 'id': 4, 'name': 'vehicle registration plate'}]}



.. code:: ipython3

    with (base_path / 'coco_dataset.json').open('w') as fp:
        json.dump(doc, fp, indent=2)
