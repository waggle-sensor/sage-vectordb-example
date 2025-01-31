import sage_data_client

# Get live lat & lon
df = sage_data_client.query(start="-5m", filter={"vsn": "W06C", "name": "sys.gps.lat|sys.gps.lon"}, tail=1)

if not df.empty:
    print(df)
    # Extract lat and lon values from the DataFrame
    lat = df[df['name'] == 'sys.gps.lat']['value'].values[0]
    lon = df[df['name'] == 'sys.gps.lon']['value'].values[0]

    # Print the dictionary
    print(float(lon))