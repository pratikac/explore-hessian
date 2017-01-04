local file = require('pl.file')
local stringx = require('pl.stringx')

local ptb = {}

function ptb.build_vocab(tokens, minfreq)
    assert(torch.type(tokens) == 'table', 'Expecting table')
    assert(torch.type(tokens[1]) == 'string', 'Expecting table of strings')
    minfreq = minfreq or -1
    assert(torch.type(minfreq) == 'number')
    local wordfreq = {}

    for i=1,#tokens do
        local word = tokens[i]
        wordfreq[word] = (wordfreq[word] or 0) + 1
    end

    local vocab, ivocab = {}, {}
    local wordseq = 0

    local _ = require 'moses'
    -- make sure ordering is consistent
    local words = _.sort(_.keys(wordfreq))

    local oov = 0
    for i, word in ipairs(words) do
        local freq = wordfreq[word]
        if freq >= minfreq then
            wordseq = wordseq + 1
            vocab[word] = wordseq
            ivocab[wordseq] = word
        else
            oov = oov + freq
        end
    end

    if oov > 0 then
        wordseq = wordseq + 1
        wordfreq['<OOV>'] = oov
        vocab['<OOV>'] = wordseq
        ivocab[wordseq] = '<OOV>'
    end

    return vocab, ivocab, wordfreq
end

function ptb.text2tensor(tokens, vocab)
    local oov = vocab['<OOV>']

    local tensor = torch.IntTensor(#tokens):fill(0)

    for i, word in ipairs(tokens) do
        local wordid = vocab[word] 
        if not wordid then
            assert(oov)
            wordid = oov
        end

        tensor[i] = wordid
    end
    return tensor
end

local loc = '/local2/pratikac/ptb'
function ptb.split(a,b)
    print('Loading PTB')

    local vocab, ivocab, wordf
    local function loadptb(f)
        local d = file.read(paths.concat(loc, f))
        d = stringx.replace(d, '\n', '<eos>')
        local tokens = stringx.split(d)
        if f == 'ptb.train.txt' then
            vocab, ivocab, wordf = ptb.build_vocab(tokens)
        end
        return ptb.text2tensor(tokens, vocab)
    end

    local train, val, test =    loadptb('ptb.train.txt'),
                                loadptb('ptb.val.txt'),
                                loadptb('ptb.test.txt')

    if opt and opt.full ~= true then
        print('Overfitting on 10% subset ...')
        local frac = 0.1
        local tn, tvn, ten = train:size(1), val:size(1), test:size(1)

        train = train:narrow(1,1,frac*tn)
        val = val:narrow(1,1,frac*tvn)
        test = test:narrow(1,1,frac*ten)
    end

    return  {data=train, size=train:size(1), vocab=vocab, ivocab=ivocab, wordf=wordf},
            {data=val, size=val:size(1)},
            {data=test, size=test:size(1)}
end

return ptb
