if __name__ == '__main__':
    import sys
    from unicon.utils import import_obj, load_obj
    from unicon.inputs import test_cb_input
    input_dev_kwds = load_obj(sys.argv[1])
    input_dev_kwds = {'type': input_dev_kwds} if isinstance(input_dev_kwds, str) else input_dev_kwds
    input_dev_type = input_dev_kwds.pop('type')
    cb_cls = import_obj(input_dev_type, default_name_prefix='cb_input', default_mod_prefix='unicon.inputs')
    test_cb_input(lambda x: cb_cls(x, **input_dev_kwds))
