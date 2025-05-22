import tables as tb

inputf  = "../nexus/kr_center.mcrd"
outputf = "../nexus/kr_center_scaled.mcrd"


with tb.open_file(inputf, "r") as fin:
    fin.copy_file(outputf, overwrite=True)

    with tb.open_file(outputf, "a") as fout:
        for i in range(2, 20):
            fout.root. pmtrd.append(fin.root. pmtrd.read()*i)
            fout.root.sipmrd.append(fin.root.sipmrd.read()*i)

            rows = []
            for row in fin.get_node("/Run/events").read():
                row = tuple(row)
                row = (i, row[1] + i)
                rows.append(row)
                print(type(row), row)
            fout.get_node("/Run/events").append(rows)

            rows = []
            for row in fin.get_node("/Run/runInfo").read():
                row = tuple(row)
                rows.append(row)
                print(type(row), row)
            fout.get_node("/Run/runInfo").append(rows)



"""
/pmtrd (EArray(5, 12, 800000)shuffle, zlib(4)) ''
/sipmrd (EArray(5, 1792, 800)shuffle, zlib(4)) ''
/Filters (Group) ''
/Filters/detected_events (Table(5,)shuffle, zlib(4)) 'Event has passed filter flag'
/Filters/signal (Table(5,)shuffle, zlib(4)) 'Event has passed filter flag'
/MC (Group) ''
/MC/configuration (Table(32,)shuffle, zlib(4)) ''
/MC/event_mapping (Table(5,)shuffle, zlib(4)) ''
/MC/hits (Table(339,)shuffle, zlib(4)) ''
/MC/particles (Table(27,)shuffle, zlib(4)) ''
/MC/sns_positions (Table(732,)shuffle, zlib(4)) ''
/MC/sns_response (Table(23675,)shuffle, zlib(4)) ''
/Run (Group) ''
/Run/eventMap (Table(5,)shuffle, zlib(4)) 'event & nexus evt for each index'
/Run/events (Table(5,)shuffle, zlib(4)) 'event info table'
/Run/runInfo (Table(5,)shuffle, zlib(4)) 'run info table'
/config (Group) ''
/config/buffy (Table(12,)shuffle, zlib(4)) 'configuration for buffy'
"""
