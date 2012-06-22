$(function() {
    tableToGrid('#parameter-list', {
        height: 550,
        pager: '#parameter-list-pager',
        rowNum: 10,
        rowList: [10,20,30]
    });
    init();
});