with open(filename) as f:
  # Split by ,
  data = []
  headers = None
  for line in f:
    line = line.split(',')
    if headers is None:
      # Read headers, which should be the first line.
      headers = line
      continue
    if len(line) != 5:
      # Unexpected.
      assert len(line) < 5, 'Number of , is less than 5... %s' %line
      line = line[:4] + [','.join(line[4:])]
    data.append(line)
  df = pd.DataFrame(data, columns=headers)
