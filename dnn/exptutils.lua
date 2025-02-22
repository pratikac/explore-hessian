local cjson = require 'cjson'

function build_file_name(opt, blacklist)
    local blacklist = blacklist or {}
    local tmp = {'backend', 'output', 'full', 'log', 'help', 'gpu',
    'max_epochs', 'batch_size', 'backend_name',
    'retrain', 'verbose', 'dataset', 'augment',
    'estimateF','rho', 'retest'}
    for k,v in pairs(tmp) do
        blacklist[#blacklist+1] = v
    end

    local _opt = torch.deserialize(torch.serialize(opt))
    for k,v in ipairs(blacklist) do _opt[v] = nil end

    local t = os.date('%b_%d_%a_%H_%M_%S')
    local s = cjson.encode(_opt)
    local fname = t .. '_opt_' .. s
    print('Log file: ' .. fname)
    return fname
end

function os.capture(cmd, raw)
    local f = assert(io.popen(cmd, 'r'))
    local s = assert(f:read('*a'))
    f:close()
    if raw then return s end
    s = string.gsub(s, '^%s+', '')
    s = string.gsub(s, '%s+$', '')
    s = string.gsub(s, '[\n\r]+', ' ')
    return s
end

function get_gitrev()
    local sha = os.capture('git rev-parse HEAD')
    local diff = os.capture('git diff', true)
    local status = os.capture('git status', true)
    return {sha=sha, diff=diff, status=status}
end

function setup_logger(opt, symbols, blacklist)
    local logger
    local fname = build_file_name(opt, blacklist)
    logger = optim.Logger(opt.output .. fname .. '.log')
    logger:setNames(symbols)

    -- inverse of symbols dict
    if not logger_stats_dict then
        logger_stats_dict = {}
        for k,v in pairs(symbols) do
            logger_stats_dict[v] = k
        end
    end
    --print(logger_stats_dict)
    return logger, fname
end

function logger_add(logger, s)
    if not logger then return end

    local t1 = {}
    for k,v in pairs(logger_stats_dict) do t1[v] = 0 end
    for k,v in pairs(s) do
        t1[logger_stats_dict[k]] = v
    end
    logger:add(t1)
end

function test_logger()
    local symbols = {'loss', 'mu', 'std', 'g', 'gmu', 'gstd'}
    local logger = setup_logger(opt, symbols)

    local s = {}
    s.loss, s.gstd = 1, 2
    logger_add(logger, s)

    s = {}
    s.std, s.gmu = 30, 20
    logger_add(logger, s)

    s = {}
    s.loss, s.mu, s.std = 4,5,6
    logger_add(logger, s)
end
