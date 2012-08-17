$(function() {
    tableToGrid('#parameter-list', {
        height: 550,
        //pager: '#parameter-list-pager',
        shrinkToFit: true,
        rowNum: -1,
        //TODO: Default sort DESC for checkbox columns
        showFilterToolbar: true,
        ignoreCase: true
    });
    init();
    $("#parameter-list").jqGrid('filterToolbar', {
        stringResult: true, 
        searchOnEnter: false,
        defaultSearch: 'cn'
        
        });
});

