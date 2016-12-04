local cjson = require 'cjson'
local tablex = require 'pl.tablex'

function build_file_name(opt, blacklist)
    local bl = tablex.union(blacklist or {},
                {'backend', 'output', 'full', 'log', 'help', 'gpu',
                'max_epochs', 'batch_size', 'backend_name',
                'retrain', 'verbose'})
    
    local _opt = tablex.deepcopy(opt)
    for k,v in ipairs(bl) do _opt[v] = nil end

    local t = os.date('%Y_%b_%d_%a_%H_%M_%S')
    local s = cjson.encode(_opt)
    local fname = t .. '_opt_' .. s
    print('Log file: ' .. fname)
    return fname
end

function setup_logger(opt, symbols)
    local logger
    local fname = build_file_name(opt)
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
