import os
import glob
import argparse

# methods = ["./audio/copysyn/ref",
#            "./audio/copysyn/wavenet",
#            "./audio/copysyn/proposed",
#            "./audio/copysyn/griffin",
#            "./audio/copysyn/anchor"
#            ]

# method_titles = ['Natural reference',
#                  'WaveNet vocoder',
#                  'Proposed',
#                  'Griffin & Lim',
#                  'Anchor'
#                  ]

methods = ["./audio/copysyn/ref",
           "./audio/copysyn/wavenet",
           "./audio/copysyn/proposed",
           "./audio/copysyn/griffin"
           ]

method_titles = ['Natural reference',
                 'WaveNet vocoder',
                 'GELP (proposed)',
                 'Griffin & Lim'
                 ]

heading =  "Copy synthesis"

def get_audio_cell(filepath):
    s = ''
    s += '<td class="audio">\n'
    s += '   <audio controls preload="none">\n'
    s += '      <source src="' + filepath + '" type="audio/wav">\n'
    s += '   </audio>\n'
    s += '</td>\n\n'
    return s

def get_row_header(i):
    s = ''
    if i%2:
        s += '</tr><tr class="odd">\n'
    else:
        s += '</tr><tr class="even">\n'
            
    s += '<td class="secondary">\n'
    s += '   Sample #' + str(i) + '\n'
    s += '</td>\n\n'
    return s

def get_main_header(text):
    s = ''
    s += '<br><br>\n'
    s += '<h2>' + text + '</h2>\n'
    s += '<br>\n\n'
    return s

def get_col_header(text):
    s = ''
    s += '<td class="audio">\n'
    s += '<h3>' + text + '</h3>\n'
    s += '</td>\n\n'
    return s

def get_audiotable_header(relative_width=100):
    s = ''
    s += '<table class ="audiotable" border="0" width="100%"><tr class="highlight">\n'
    s += '<tr class="highlight">\n'
    s += '<td class="secondary">\n'
    s += '</td>\n\n' 
    return s

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default="./audiotable.html")
    parser.add_argument('--max_files', type=int, default=20)
    return parser.parse_args(argv)

if __name__ == "__main__":

    fnames = glob.glob(methods[0] + '/*.wav')
     
    args = parse_args()

    s = '<div class="visible" id="area2">\n\n'
    s += get_main_header(heading)
    s += get_audiotable_header()

    assert len(methods) == len(method_titles)

    for m in method_titles:
        s += get_col_header(m)

    for i, fname in enumerate(fnames):
       
        bname = os.path.basename(fname)
        s += get_row_header(i+1)
 
        for m in methods:
            mfname = os.path.join(m, bname)
            if not os.path.exists(mfname):
                bname_noext = os.path.splitext(bname)[0]
                alt_mfname = os.path.join(m, bname_noext + '.syn.wav')
                if os.path.isfile(alt_mfname):
                    mfname = alt_mfname
                else:    
                    print('Warning: file "' + mfname + '" not found')
            s += get_audio_cell(mfname)

        if i >= args.max_files-1:
            break

    s += '</tr>\n'

    s += '</table>\n\n'
    s += '<br><br>\n\n'
    s += '</div>\n'

    out_file = open(args.output, 'w')
    out_file.write(s)
    out_file.close()
